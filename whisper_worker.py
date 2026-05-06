# whisper_worker.py
import os
import sys
import json
import gc
import time

# Force Mac stability
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_DEBUG_CPU_TYPE"] = "5"

from faster_whisper import WhisperModel
import shutil

# Resolve system binaries (kept for parity with other workers)
FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE_BIN = shutil.which("ffprobe") or "ffprobe"


def run_transcription(audio_path, model_size, output_json):
    # Use int8 if available to reduce memory footprint on CPU
    print(f"Loading Whisper {model_size} on cpu (try int8)...")
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)
    except Exception:
        print("int8 not supported, falling back to float32")
        model = WhisperModel(model_size, device="cpu", compute_type="float32", cpu_threads=4)

    print("Transcribing (streaming segments)...")
    segments_gen, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="de",
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    words = []
    last_report = 0.0
    total_duration = float(getattr(info, "duration", 0.0) or 0.0)
    for seg in segments_gen:
        # iterate streaming segments to reduce peak memory
        if getattr(seg, "words", None):
            for word in seg.words:
                words.append({
                    "start": word.start,
                    "end": word.end,
                    "word": word.word.strip(),
                    "raw_token": word.word,
                })

        # print progress by last segment end
        try:
            progress = getattr(seg, "end", last_report)
        except Exception:
            progress = last_report
        if progress and (progress - last_report) >= 30.0:  # every 30s of audio processed
            total_display = f"/{int(total_duration)}s" if total_duration else ""
            print(f"  ASR progress: processed ~{int(progress)}s{total_display}")
            last_report = progress

    # final write
    with open(output_json, "w") as f:
        json.dump(words, f)

    # free memory
    try:
        del model
    except Exception:
        pass
    gc.collect()


if __name__ == "__main__":
    run_transcription(sys.argv[1], sys.argv[2], sys.argv[3])