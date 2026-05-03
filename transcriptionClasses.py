from typing import List, TypedDict, Any
from typing_extensions import NotRequired


class Segment(TypedDict):
    id: int
    start: float
    end: float
    text: str
    speaker: NotRequired[str]


class TranscriptionResult(TypedDict):
    task: str
    language: str
    duration: float
    model: str
    text: str
    segments: List[Segment]
    words: List[dict[str, Any]]
    audio_duration: NotRequired[float]
    preprocessing_time: NotRequired[float]
    transcription_time: NotRequired[float]
    diarization_time: NotRequired[float]
    handoff_time: NotRequired[float]
    total_time: NotRequired[float]
    transcription_rtf: NotRequired[float]
    diarization_rtf: NotRequired[float]
    total_rtf: NotRequired[float]
    runtime: NotRequired[float]
