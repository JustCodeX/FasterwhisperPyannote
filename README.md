# fasterwhisperpyannote

Pipeline zur automatischen Transkription und Sprecherdiarisierung von Audio-Dateien mit `faster-whisper` und `pyannote.audio`.

## Ziel

Dieses Projekt verarbeitet Audio-Dateien end-to-end:

1. Audio wird zuerst auf ein internes 16 kHz Mono-WAV-Format normalisiert.
2. `faster-whisper` erzeugt Wort-/Segment-Transkripte.
3. `pyannote.audio` erkennt Sprecherwechsel und erzeugt Diarisierungs-Turns.
4. Die Ergebnisse werden zusammengeführt und als Transkript mit Sprecherzuordnung gespeichert.

Die aktuelle Implementierung ist auf stabile Ausführung bei längeren Dateien ausgelegt. Große Audio-Dateien werden bei Bedarf in Chunks verarbeitet, damit ASR und Diarisierung nicht gleichzeitig unnötig viel Speicher belegen.

## Projektstruktur

- `main_pipline.py` - Orchestriert die komplette Verarbeitung.
- `whisper_worker.py` - Führt die ASR-Transkription mit `faster-whisper` aus.
- `pyannote_worker.py` - Führt die Diarisierung mit `pyannote.audio` aus.
- `eval.py` - Lädt Ground-Truth-Daten und berechnet Metriken wie WER, CER, DER und WDER.
- `savetranscript.py` - Erstellt und speichert menschenlesbare Ergebnisberichte.
- `transcriptionClasses.py` - Typdefinitionen für Segmente und Transkriptionsergebnisse.
- `audiofiles/` - Eingabedateien für lokale Experimente.
- `groundtruth/` - Referenzdaten für die Auswertung.
- `Transcriptions/` - Ausgabeordner für gespeicherte Transkripte.

## Pipeline

### 1. Audio-Preprocessing

Die Eingabedatei wird per `ffmpeg` auf 16 kHz Mono normalisiert. Das reduziert Formatprobleme bei MP3s und macht die nachfolgenden Modelle robuster.

### 2. Whisper-ASR

`whisper_worker.py` lädt `faster-whisper` und transkribiert das Audio mit Wort-Timestamps. Für lange Dateien wird die Verarbeitung progressiv ausgeführt, damit der Speicherverbrauch niedrig bleibt.

### 3. Pyannote-Diarisierung

`pyannote_worker.py` lädt das Diarisierungsmodell und bestimmt Sprecher-Turns. Für lange Audiodateien kann die Diarisierung in überlappenden Chunks laufen, damit die Berechnung nicht hängen bleibt.

### 4. Zusammenführen der Ergebnisse

Die Wort- und Sprecherinformationen werden zusammengeführt. Danach werden Segmente aufgebaut und ein vollständiger Text sowie Sprecherzuordnungen erzeugt.

### 5. Bewertung und Export

Optional können Ground-Truth-Daten geladen werden. Daraus werden Metriken wie WER, CER, DER und WDER berechnet. Die finale Ausgabe wird anschließend als Textdatei im `Transcriptions/`-Ordner gespeichert.

## Wichtige Implementierungsentscheidungen

- Verarbeitung von ASR und Diarisierung nacheinander, nicht gleichzeitig.
- Explizites Freigeben von Modellen und Cache nach den einzelnen Schritten.
- Chunking für lange Audiodateien, um Speicherprobleme auf älteren GPUs und Macs zu vermeiden.
- Temporäre WAV-Dateien werden nach der Verarbeitung wieder gelöscht.
- Konsistente Sprecherlabels und Segmentgrenzen werden nachträglich normalisiert.

## Voraussetzungen

- Python 3.10+.
- `ffmpeg` und `ffprobe` im PATH.
- Zugriff auf Hugging Face Token für `pyannote.audio`.
- Eine GPU ist hilfreich, aber die Pipeline kann auch auf CPU laufen.

## Installation

Beispiel:

```bash
pip install faster-whisper pyannote.audio huggingface_hub torch soundfile jiwer
```

Falls du eine eigene Umgebung nutzt, installiere zusätzlich die jeweiligen systemnahen Abhängigkeiten wie `ffmpeg`.

## Ausführen

Einzeldatei-Verarbeitung:

```bash
python main_pipline.py /path/to/audio.mp3
```

Mit explizitem Modell:

```bash
python main_pipline.py /path/to/audio.mp3 --model TheChola/whisper-large-v3-turbo-german-faster-whisper
```

Mit Referenzdatei für die Auswertung:

```bash
python main_pipline.py /path/to/audio.mp3 --ground-truth /path/to/file.rttm
```

## Hinweise für Git

Die folgenden Inhalte sollten nicht ins Repository:

- erzeugte Transkripte
- temporäre JSON-/WAV-Dateien
- Modell-Caches
- virtuelle Umgebungen
- Log-Dateien

Diese werden über `.gitignore` ausgeschlossen.

## Status

Das Projekt ist aktuell so aufgebaut, dass es direkt für einen Git-Upload vorbereitet werden kann: Quellcode, Dokumentation und Ignore-Regeln sind getrennt, und nur die eigentliche Implementierung soll versioniert werden.