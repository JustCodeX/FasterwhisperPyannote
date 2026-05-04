import re
from pathlib import Path
from typing import Any, TypedDict, cast

from jiwer import cer, wer

from transcriptionClasses import TranscriptionResult


class GroundTruthTurn(TypedDict):
    start: float
    end: float
    speaker: str
    text: str


_TXT_GROUND_TRUTH_LINE = re.compile(
    r"^\[(?P<start>[0-9.]+)s\s+--?\s+(?P<end>[0-9.]+)s\]\s+(?P<speaker>SPEAKER_\d+):\s+(?P<text>.*)$"
)
_RTTM_LINE = re.compile(
    r"^SPEAKER\s+\S+\s+\d+\s+(?P<start>[0-9.]+)\s+(?P<duration>[0-9.]+)\s+<NA>\s+<NA>\s+(?P<speaker>\S+)"
)


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    return wer(_normalize_text(reference), _normalize_text(hypothesis))


def compute_cer(reference: str, hypothesis: str) -> float:
    return cer(_normalize_text(reference), _normalize_text(hypothesis))


def _get_ground_truth_path(audio_file: str) -> Path:
    """Get path to RTTM ground truth file."""
    audio_path = Path(audio_file).expanduser().resolve()
    for parent in [audio_path.parent, *audio_path.parents]:
        candidate = parent / "groundtruth" / f"{audio_path.stem}.rttm"
        if candidate.exists():
            return candidate
    return audio_path.parent / "groundtruth" / f"{audio_path.stem}.rttm"


def _get_ground_truth_txt_path(audio_file: str) -> Path:
    """Get path to TXT ground truth file."""
    audio_path = Path(audio_file).expanduser().resolve()
    for parent in [audio_path.parent, *audio_path.parents]:
        candidate = parent / "groundtruth" / f"{audio_path.stem}.txt"
        if candidate.exists():
            return candidate
    return audio_path.parent / "groundtruth" / f"{audio_path.stem}.txt"


def _parse_rttm_file(rttm_path: Path) -> dict[str, Any]:
    """Parse RTTM file for diarization turns."""
    if not rttm_path.exists():
        return {"turns": [], "path": str(rttm_path)}

    turns: list[GroundTruthTurn] = []
    for line in rttm_path.read_text(encoding="utf-8").splitlines():
        match = _RTTM_LINE.match(line.strip())
        if not match:
            continue
        start = float(match.group("start"))
        duration = float(match.group("duration"))
        turns.append(
            {
                "start": start,
                "end": start + duration,
                "speaker": match.group("speaker").strip(),
                "text": "",
            }
        )

    return {
        "turns": turns,
        "path": str(rttm_path),
    }


def _parse_txt_file(txt_path: Path) -> dict[str, Any]:
    """Parse TXT file for transcription text."""
    if not txt_path.exists():
        return {"text": "", "turns": [], "path": str(txt_path)}

    turns: list[GroundTruthTurn] = []
    full_text_parts: list[str] = []

    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        match = _TXT_GROUND_TRUTH_LINE.match(line)
        if not match:
            full_text_parts.append(line)
            continue
        turns.append(
            {
                "start": float(match.group("start")),
                "end": float(match.group("end")),
                "speaker": match.group("speaker").strip(),
                "text": match.group("text").strip(),
            }
        )
        full_text_parts.append(match.group("text").strip())

    return {
        "text": " ".join(full_text_parts).strip(),
        "turns": turns,
        "path": str(txt_path),
    }


def _parse_ground_truth_file(ground_truth_path: Path) -> dict[str, Any]:
    if not ground_truth_path.exists():
        return {"text": "", "turns": [], "path": str(ground_truth_path)}

    turns: list[GroundTruthTurn] = []
    for line in ground_truth_path.read_text(encoding="utf-8").splitlines():
        match = _TXT_GROUND_TRUTH_LINE.match(line.strip())
        if not match:
            continue
        turns.append(
            {
                "start": float(match.group("start")),
                "end": float(match.group("end")),
                "speaker": match.group("speaker").strip(),
                "text": match.group("text").strip(),
            }
        )

    return {
        "text": " ".join(turn["text"] for turn in turns).strip(),
        "turns": turns,
        "path": str(ground_truth_path),
    }


def load_ground_truth(audio_file: str) -> dict[str, Any]:
    """Load ground truth from both TXT (transcription) and RTTM (diarization) files."""
    txt_data = _parse_txt_file(_get_ground_truth_txt_path(audio_file))
    rttm_data = _parse_rttm_file(_get_ground_truth_path(audio_file))
    
    return {
        "text": txt_data["text"],
        "turns": rttm_data["turns"],
        "txt_path": txt_data["path"],
        "rttm_path": rttm_data["path"],
    }


def _speaker_at_time(turns: list[GroundTruthTurn], timestamp: float) -> str:
    for turn in turns:
        if turn["start"] <= timestamp <= turn["end"]:
            return turn["speaker"]
    return "unknown"


def compute_der(reference_turns: list[GroundTruthTurn], hypothesis_segments: list[dict[str, Any]]) -> float:
    if not reference_turns:
        return 0.0

    boundaries = {turn["start"] for turn in reference_turns} | {turn["end"] for turn in reference_turns}
    for segment in hypothesis_segments:
        boundaries.add(float(segment.get("start", 0.0)))
        boundaries.add(float(segment.get("end", 0.0)))

    ordered = sorted(boundaries)
    if len(ordered) < 2:
        return 0.0

    reference_total = sum(max(0.0, turn["end"] - turn["start"]) for turn in reference_turns)
    error = 0.0

    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            continue
        midpoint = (left + right) / 2.0
        reference_speaker = _speaker_at_time(reference_turns, midpoint)
        if reference_speaker == "unknown":
            continue

        hypothesis_speaker = "unknown"
        for segment in hypothesis_segments:
            segment_start = float(segment.get("start", 0.0))
            segment_end = float(segment.get("end", 0.0))
            if segment_start <= midpoint <= segment_end:
                hypothesis_speaker = str(segment.get("speaker", "unknown"))
                break

        if hypothesis_speaker != reference_speaker:
            error += right - left

    return error / reference_total if reference_total > 0.0 else 0.0


def compute_wder(reference_turns: list[GroundTruthTurn], hypothesis_segments: list[dict[str, Any]]) -> float:
    if not reference_turns:
        return 0.0

    total_words = 0
    wrong_words = 0

    for segment in hypothesis_segments:
        text = str(segment.get("text", "")).strip()
        words = [word for word in text.split() if word]
        if not words:
            continue

        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        duration = max(0.0, end - start)
        step = duration / len(words) if words else 0.0
        speaker = str(segment.get("speaker", "unknown"))

        for index in range(len(words)):
            total_words += 1
            midpoint = start + (index + 0.5) * step if step > 0.0 else (start + end) / 2.0
            reference_speaker = _speaker_at_time(reference_turns, midpoint)
            if reference_speaker == "unknown" or speaker != reference_speaker:
                wrong_words += 1

    return wrong_words / total_words if total_words > 0 else 0.0


def evaluate_result(
    result: TranscriptionResult,
    ground_truth_text: str,
    reference_turns: list[GroundTruthTurn],
) -> dict[str, float]:
    hypothesis = str(result.get("text", ""))
    segments = cast(list[dict[str, Any]], result.get("segments", []))
    return {
        "wer": compute_wer(ground_truth_text, hypothesis),
        "cer": compute_cer(ground_truth_text, hypothesis),
        "der": compute_der(reference_turns, segments),
        "wder": compute_wder(reference_turns, segments),
    }
