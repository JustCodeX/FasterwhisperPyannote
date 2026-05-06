import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import math
import gc
from pathlib import Path
from typing import Any, Dict, List, cast
import warnings
import sys
import shutil

warnings.filterwarnings("ignore", category=UserWarning)

from huggingface_hub import get_token
from eval import GroundTruthTurn, evaluate_result, load_ground_truth, _parse_ground_truth_file
from savetranscript import save_transcription_to_file
from transcriptionClasses import Segment, TranscriptionResult

try:
    import torch
except Exception:
    torch = None

def _safe_rtf(process_time: float, audio_duration: float) -> float:
    if audio_duration <= 0: return 0.0
    return process_time / audio_duration

DEFAULT_MODEL = "TheChola/whisper-large-v3-turbo-german-faster-whisper"

# resolve system binaries (ffmpeg / ffprobe) if available on PATH
FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE_BIN = shutil.which("ffprobe") or "ffprobe"

def _clean_text(text: str):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    return text.strip()

def _build_text_from_words(words):
    token_stream = "".join(str(w.get("raw_token", w.get("word", ""))) for w in words)
    return _clean_text(token_stream)

def _format_speaker_label(raw_label):
    m = re.search(r"(\d+)$", str(raw_label))
    return f"SPEAKER_{int(m.group(1)):02d}" if m else str(raw_label)


def _preprocess_audio_with_ffmpeg(audio_path: str) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix="_converted.wav")
    temp_file.close()
    output_path = temp_file.name
    print("Converting audio to 16kHz mono WAV...")
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        audio_path,
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg conversion failed: {exc.stderr.decode()}")


def _get_audio_duration_seconds_ffprobe(audio_path: str) -> float:
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return float(out)
    except Exception:
        return 0.0


def _split_audio_into_chunks(input_wav: str, chunk_seconds: int = 600) -> List[str]:
    duration = _get_audio_duration_seconds_ffprobe(input_wav)
    if duration <= 0 or duration <= chunk_seconds:
        return [input_wav]

    out_dir = tempfile.mkdtemp(prefix="audio_chunks_")
    chunk_paths: List[str] = []
    start = 0.0
    idx = 0
    while start < duration - 0.001:
        out_path = os.path.join(out_dir, f"chunk_{idx:03d}.wav")
        cmd = [
            FFMPEG_BIN,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            input_wav,
            "-ss",
            str(math.floor(start)),
            "-t",
            str(chunk_seconds),
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            out_path,
        ]
        subprocess.run(cmd, check=True)
        chunk_paths.append(out_path)
        idx += 1
        start += chunk_seconds

    return chunk_paths

def preprocess_turns(turns: List[Dict], merge_threshold=0.5):
    if not turns: return []
    for t in turns: t["speaker"] = _format_speaker_label(t["speaker"])
    
    smoothed = []
    curr = turns[0].copy()
    for i in range(1, len(turns)):
        nxt = turns[i]
        if nxt["speaker"] == curr["speaker"] and (nxt["start"] - curr["end"]) < merge_threshold:
            curr["end"] = nxt["end"]
        else:
            smoothed.append(curr)
            curr = nxt.copy()
    smoothed.append(curr)
    
    final_turns = [t for t in smoothed if (t["end"] - t["start"]) > 0.3]
    return final_turns

def assign_speaker_to_words(words, turns):
    turns = preprocess_turns(turns)

    if not turns:
        for word in words:
            word["speaker"] = "SPEAKER_00"
        return words

    for word in words:
        word_start = float(word["start"])
        word_end = float(word["end"])
        overlaps: List[tuple[float, str]] = []

        for turn in turns:
            overlap_start = max(word_start, float(turn["start"]))
            overlap_end = min(word_end, float(turn["end"]))
            overlap_time = max(0.0, overlap_end - overlap_start)
            if overlap_time > 0.0:
                overlaps.append((overlap_time, str(turn["speaker"])))

        if overlaps:
            word["speaker"] = max(overlaps, key=lambda item: item[0])[1]
        else:
            midpoint = (word_start + word_end) / 2.0
            closest = min(
                turns,
                key=lambda turn: min(abs(midpoint - float(turn["start"])), abs(midpoint - float(turn["end"]))),
            )
            word["speaker"] = str(closest["speaker"])

    for i in range(1, len(words)):
        if words[i].get("raw_token") in [".", ",", "!", "?", " .", " ,"]:
            words[i]["speaker"] = words[i - 1]["speaker"]

    for i in range(1, len(words) - 1):
        prev_speaker = words[i - 1]["speaker"]
        next_speaker = words[i + 1]["speaker"]
        if prev_speaker == next_speaker and words[i]["speaker"] != prev_speaker:
            words[i]["speaker"] = prev_speaker

    return words

def group_words_into_segments(words) -> List[Segment]:
    if not words: return []
    segments = []
    current_words = [words[0]]
    curr_speaker = words[0].get("speaker")
    
    for i in range(1, len(words)):
        w = words[i]
        # Break if speaker changes OR if there's a significant silence gap (> 1.2s)
        gap = w["start"] - words[i-1]["end"]
        
        if w["speaker"] != curr_speaker or gap > 1.2:
            segments.append({
                "id": len(segments),
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
                "text": _build_text_from_words(current_words),
                "speaker": curr_speaker
            })
            current_words, curr_speaker = [w], w["speaker"]
        else:
            current_words.append(w)
            
    segments.append({
        "id": len(segments),
        "start": current_words[0]["start"],
        "end": current_words[-1]["end"],
        "text": _build_text_from_words(current_words),
        "speaker": curr_speaker
    })
    return cast(List[Segment], segments)

def run_pipeline(audio_path, model_id: str = DEFAULT_MODEL, num_speakers: int = 2) -> TranscriptionResult:
    start_total = time.perf_counter()
    hf_token = os.getenv("HF_TOKEN") or get_token() or ""
    audio_abs = str(Path(audio_path).resolve())
    words_tmp, diar_tmp = "words_tmp.json", "diar_tmp.json"

    # ensure ffmpeg/ffprobe are available and the Python interpreter path is explicit
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffmpeg and/or ffprobe not found in PATH.\n"
            "Install with 'brew install ffmpeg' on macOS or 'sudo apt install ffmpeg' on Debian/Ubuntu.\n"
            "After installing, ensure they are available in your PATH and retry."
        )

    preprocessing_start = time.perf_counter()
    converted_path = _preprocess_audio_with_ffmpeg(audio_abs)
    preprocessing_time = time.perf_counter() - preprocessing_start

    audio_duration = _get_audio_duration_seconds_ffprobe(converted_path)

    # If audio too long, split into 10-minute chunks for Whisper to keep memory low
    chunk_paths = [converted_path]
    if audio_duration > 15 * 60:
        print("Audio >15 minutes: splitting into 10-minute chunks for ASR to reduce memory footprint.")
        chunk_paths = _split_audio_into_chunks(converted_path, chunk_seconds=10 * 60)

    aggregated_words: List[dict] = []
    # Run Whisper on chunks sequentially to avoid having both models in memory
    t_start_asr = time.perf_counter()
    for idx, chunk in enumerate(chunk_paths, start=1):
        print(f"Processing ASR chunk {idx}/{len(chunk_paths)}: {os.path.basename(chunk)}")
        chunk_words_tmp = f"words_tmp_{idx}.json"
        # invoke worker with the same Python interpreter to ensure environment consistency
        whisper_script = Path(__file__).resolve().parent / "whisper_worker.py"
        subprocess.run([sys.executable, str(whisper_script), chunk, model_id, chunk_words_tmp], check=True)

        # read and offset times by chunk start
        try:
            with open(chunk_words_tmp, 'r') as f:
                chunk_words = json.load(f)
        except Exception:
            chunk_words = []

        # determine chunk offset from filename by probing its start time using ffprobe
        # we created chunks via -ss so their times are relative to 0, so find offset by (idx-1)*chunk_seconds
        offset = 0.0
        if len(chunk_paths) > 1:
            offset = (idx - 1) * (10 * 60)

        for w in chunk_words:
            w["start"] = float(w.get("start", 0.0)) + offset
            w["end"] = float(w.get("end", 0.0)) + offset
            aggregated_words.append(w)

        # cleanup chunk file
        if os.path.exists(chunk_words_tmp):
            os.remove(chunk_words_tmp)

        # encourage memory free between subprocesses
        gc.collect()
        try:
            if torch is not None and hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    asr_time = time.perf_counter() - t_start_asr

    # Write aggregated words to words_tmp for downstream steps
    with open(words_tmp, 'w') as f:
        json.dump(aggregated_words, f)

    # Now run diarization once (sequential after ASR complete)
    t_start_diar = time.perf_counter()
    pyannote_script = Path(__file__).resolve().parent / "pyannote_worker.py"
    subprocess.run([sys.executable, str(pyannote_script), converted_path, str(num_speakers), diar_tmp, hf_token], check=True)
    diar_time = time.perf_counter() - t_start_diar

    with open(words_tmp, 'r') as f:
        words = json.load(f)
    with open(diar_tmp, 'r') as f:
        turns = json.load(f)

    processed_words = assign_speaker_to_words(words, turns)
    segments = group_words_into_segments(processed_words)
    
    full_text = " ".join([s["text"] for s in segments])
    audio_duration = turns[-1]["end"] if turns else 0.0
    total_time = time.perf_counter() - start_total

    result: TranscriptionResult = {
        "task": "transcribe", "language": "de", "duration": audio_duration,
        "model": model_id, "text": full_text, "segments": segments, "words": processed_words,
        "audio_duration": audio_duration, "preprocessing_time": preprocessing_time,
        "transcription_time": asr_time,
        "diarization_time": diar_time, "total_time": total_time, "runtime": total_time,
        "handoff_time": 0.0,
        "transcription_rtf": _safe_rtf(asr_time, audio_duration),
        "diarization_rtf": _safe_rtf(diar_time, audio_duration),
        "total_rtf": _safe_rtf(total_time, audio_duration),
        "preprocessing_rtf": _safe_rtf(preprocessing_time, audio_duration),
    }

    for f in [words_tmp, diar_tmp]:
        if os.path.exists(f):
            os.remove(f)
    # remove chunk dirs if any
    if len(chunk_paths) > 1:
        base_dir = os.path.dirname(chunk_paths[0])
        try:
            for fn in os.listdir(base_dir):
                if fn.startswith("chunk_") and fn.endswith('.wav'):
                    os.remove(os.path.join(base_dir, fn))
            os.rmdir(base_dir)
        except Exception:
            pass

    if os.path.exists(converted_path):
        os.remove(converted_path)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using faster-whisper and pyannote")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--ground-truth", default=None, help="Optional explicit path to an RTTM ground-truth file")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--response-format", default="text", choices=["json", "verbose_json", "text"])
    args = parser.parse_args()

    result = run_pipeline(args.audio_file, model_id=args.model, num_speakers=args.num_speakers)

    ground_truth_data = (
        _parse_ground_truth_file(Path(args.ground_truth).expanduser().resolve())
        if args.ground_truth
        else load_ground_truth(args.audio_file)
    )

    metrics = evaluate_result(
        result,
        ground_truth_data.get("text", ""),
        ground_truth_data.get("turns", cast(list[GroundTruthTurn], [])),
    )
    result["ground_truth"] = ground_truth_data.get("text", "")
    result["wer"] = metrics["wer"]
    result["cer"] = metrics["cer"]
    result["der"] = metrics["der"]
    result["wder"] = metrics["wder"]
    result["audio_file"] = args.audio_file

    save_transcription_to_file(result)

    print()
    print("=== FINAL RESULTS SUMMARY ===")
    print(
        f"Model: {result['model']} | WER: {metrics['wer']:.3f} | CER: {metrics['cer']:.3f} | "
        f"DER: {metrics['der']:.3f} | WDER: {metrics['wder']:.3f} | Runtime: {result.get('runtime', 0.0):.2f}s"
    )

    if args.response_format in {"json", "verbose_json"}:
        print(json.dumps(result, indent=2))