import re
from pathlib import Path
from typing import cast

import soundfile as sf

from transcriptionClasses import Segment, TranscriptionResult


def _get_audio_duration_seconds(audio_file: str, result: TranscriptionResult) -> float:
    try:
        return float(sf.info(audio_file).duration)
    except Exception:
        segments = result.get("segments", [])
        if not segments:
            return 0.0
        return max(float(seg.get("end", 0.0)) for seg in segments)


def _safe_rtf(step_time: float, audio_duration: float) -> float:
    if audio_duration <= 0.0:
        return 0.0
    return step_time / audio_duration


def _safe_filename_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "unknown"


def _bundle_segments(segments: list[Segment]) -> list[Segment]:
    """Merge consecutive segments from the same speaker into one segment."""
    if not segments:
        return segments

    bundled: list[Segment] = []
    current_bundle = dict(segments[0])

    for i in range(1, len(segments)):
        next_segment = segments[i]
        if next_segment.get("speaker") == current_bundle.get("speaker"):
            # Same speaker: merge text and extend end time
            current_text = current_bundle.get("text", "").strip()
            next_text = next_segment.get("text", "").strip()
            current_bundle["text"] = f"{current_text} {next_text}".strip()
            current_bundle["end"] = next_segment.get("end", current_bundle.get("end"))
        else:
            # Different speaker: save current bundle and start new one
            bundled.append(current_bundle)
            current_bundle = dict(next_segment)

    bundled.append(current_bundle)
    return bundled


def build_transcription_report(result: TranscriptionResult) -> list[str]:
    audio_file = str(result.get("audio_file", "unknown_audio"))
    model_size = str(result.get("model", "unknown_model"))

    runtime = float(result.get("runtime", 0.0))
    preprocessing_time = float(result.get("preprocessing_time", 0.0))
    transcription_time = float(result.get("transcription_time", 0.0))
    diarization_time = float(result.get("diarization_time", 0.0))
    audio_duration = float(result.get("audio_duration", 0.0)) or _get_audio_duration_seconds(audio_file, result)

    rtf = _safe_rtf(runtime, audio_duration)
    rtf_preprocessing = _safe_rtf(preprocessing_time, audio_duration)
    rtf_transcription = _safe_rtf(transcription_time, audio_duration)
    rtf_diarization = _safe_rtf(diarization_time, audio_duration)
    segments = cast(list[Segment], result.get("segments", []))
    speakers = sorted({str(seg.get("speaker", "unknown")) for seg in segments})
    has_real_diarization = any(speaker != "unknown" for speaker in speakers)
    ground_truth_text = str(result.get("ground_truth", "N/A")).strip()
    der = float(result.get("der", 0.0))
    wder = float(result.get("wder", 0.0))

    lines = [
        "TRANSCRIPTION RESULTS",
        "=" * 50,
        "",
        f"Audio File: {audio_file}",
        f"Model: {model_size}",
        f"Language: {result.get('language', 'unknown')}",
        "",
        f"Runtime: {runtime:.2f} seconds",
        f"  - Preprocessing: {preprocessing_time:.2f} seconds (RTF: {rtf_preprocessing:.3f})",
        f"  - Transcription: {transcription_time:.2f} seconds (RTF: {rtf_transcription:.3f})",
        f"  - Diarization: {diarization_time:.2f} seconds (RTF: {rtf_diarization:.3f})",
        f"Audio Duration: {audio_duration:.2f} seconds",
        f"Diarization: {'yes' if has_real_diarization else 'no'}",
        f"Speakers: {', '.join(speakers) if speakers else 'unknown'}",
        f"Total RTF: {rtf:.3f}",
        "",
        "Full Text:",
        f"{result.get('text', '').strip()}",
        "",
        "Detailed Segments:",
        "-" * 40,
    ]

    bundled_segments = _bundle_segments(segments)
    for segment in bundled_segments:
        start_time = f"{segment.get('start', 0.0)}"
        end_time = f"{segment.get('end', 0.0)}"
        text = segment.get("text", "").strip()
        speaker = segment.get("speaker", "unknown")
        lines.append(f"Speaker {speaker}: [{start_time}s --> {end_time}s] {text}")

    lines.extend(
        [
            "-" * 40,
            "Ground Truth:",
            f"{ground_truth_text}",
            f"WER: {result.get('wer', 0.0):.3f}",
            f"CER: {result.get('cer', 0.0):.3f}",
            f"DER: {der:.3f}",
            f"WDER: {wder:.3f}",
            f"RTF (Total): {rtf:.3f}",
            f"RTF (Preprocessing): {rtf_preprocessing:.3f}",
            f"RTF (Transcription only): {rtf_transcription:.3f}",
            f"RTF (Diarization only): {rtf_diarization:.3f}",
        ]
    )
    return lines


def print_transcription_report(result: TranscriptionResult) -> None:
    print()
    for line in build_transcription_report(result):
        print(line)


def save_transcription_to_file(result: TranscriptionResult) -> None:
    audio_file = str(result.get("audio_file", "unknown_audio"))
    model_size = _safe_filename_token(str(result.get("model", "unknown_model")))

    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"Audio file not found: {audio_file}")
        return

    try:
        base_dir = Path(__file__).resolve().parent
        output_dir = base_dir / "Transcriptions"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{audio_path.stem}_transcription_{model_size}.txt"
        report_lines = build_transcription_report(result)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")

        print(f"Transcription saved to: {output_file}")
    except Exception as exc:
        print(f"Error: {exc}")
