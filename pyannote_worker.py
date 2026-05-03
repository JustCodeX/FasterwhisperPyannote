# pyannote_worker.py
import os
import sys
import json
import subprocess
import tempfile
import math
import gc
import warnings
from typing import List

warnings.filterwarnings("ignore", category=UserWarning)

from pyannote.audio import Pipeline
try:
    import torch
except Exception:
    torch = None


def _get_audio_duration_seconds_ffprobe(audio_path: str) -> float:
    cmd = [
        "ffprobe",
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


def _split_audio_for_diarization(input_wav: str, chunk_seconds: int = 600, overlap: int = 2) -> List[tuple[str, float]]:
    duration = _get_audio_duration_seconds_ffprobe(input_wav)
    if duration <= 0 or duration <= chunk_seconds:
        return [(input_wav, 0.0)]

    chunks = []
    start = 0.0
    idx = 0
    out_dir = tempfile.mkdtemp(prefix="diar_chunks_")
    while start < duration - 0.001:
        out_path = os.path.join(out_dir, f"diar_chunk_{idx:03d}.wav")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            input_wav,
            "-ss",
            str(max(0, math.floor(start - overlap))),
            "-t",
            str(chunk_seconds + overlap),
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
        chunks.append((out_path, start))
        idx += 1
        start += chunk_seconds

    return chunks


def _merge_and_normalize_turns(all_turns: List[dict]) -> List[dict]:
    if not all_turns:
        return []
    # sort and merge adjacent turns with same speaker when gap small
    turns = sorted(all_turns, key=lambda t: t["start"])
    merged = [turns[0].copy()]
    for t in turns[1:]:
        last = merged[-1]
        # if same speaker and gap small, extend
        gap = t["start"] - last["end"]
        if t["speaker"] == last["speaker"] and gap <= 0.5:
            last["end"] = max(last["end"], t["end"])
        else:
            merged.append(t.copy())
    # round times
    for m in merged:
        m["start"] = round(float(m["start"]), 3)
        m["end"] = round(float(m["end"]), 3)
    return merged


def run_diarization(audio_path, num_speakers, output_json, hf_token):
    print("Loading pyannote diarization pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=hf_token)

    # move pipeline to MPS if available (Mac), otherwise CPU
    try:
        device = "mps" if torch is not None and torch.backends.mps.is_available() else "cpu"
        pipeline.to(device)
    except Exception:
        pass

    num_speakers = int(num_speakers) if num_speakers != "None" else None

    duration = _get_audio_duration_seconds_ffprobe(audio_path)
    print(f"Diarization: audio duration {int(duration)}s")

    all_turns = []
    # chunk if long
    if duration > 600:
        print("Long audio detected: running diarization in chunks and merging results.")
        chunks = _split_audio_for_diarization(audio_path, chunk_seconds=600, overlap=2)
        for idx, (chunk_path, offset) in enumerate(chunks, start=1):
            print(f"  Diarization chunk {idx}/{len(chunks)} (offset {int(offset)}s)")
            diarization = pipeline(chunk_path, num_speakers=num_speakers)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                all_turns.append({
                    "start": float(turn.start) + offset,
                    "end": float(turn.end) + offset,
                    "speaker": str(speaker),
                })
            try:
                os.remove(chunk_path)
            except Exception:
                pass
    else:
        print("Diarizing whole file...")
        diarization = pipeline(audio_path, num_speakers=num_speakers)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            all_turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})

    merged = _merge_and_normalize_turns(all_turns)

    with open(output_json, "w") as f:
        json.dump(merged, f)

    # free memory
    try:
        del pipeline
    except Exception:
        pass
    gc.collect()
    try:
        if torch is not None and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    run_diarization(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])