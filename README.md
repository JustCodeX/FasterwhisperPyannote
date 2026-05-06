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
Empfohlener, reproduzierbarer Weg (virtuelle Umgebung):

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Wichtig: `ffmpeg`/`ffprobe` müssen als native Systempakete installiert sein. Beispiele:

```bash
# macOS (Homebrew)
brew install ffmpeg

# Debian/Ubuntu
sudo apt update && sudo apt install -y ffmpeg
```

Setze optional `HF_TOKEN` in der Umgebung, z.B. `export HF_TOKEN="<your_token>"`.

Quick start (automated):

```bash
# Make the helper script executable (first time)
chmod +x scripts/setup.sh
# Run setup (creates .venv and installs pinned requirements)
./scripts/setup.sh
```

The `scripts/setup.sh` helper checks for `ffmpeg`/`ffprobe`, creates a virtual environment, and installs the pinned packages from `requirements.txt`.

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

## Troubleshooting

- **ffmpeg/ffprobe nicht gefunden**
  ```bash
  which ffmpeg && which ffprobe
  # Wenn nicht vorhanden:
  # macOS: brew install ffmpeg
  # Debian/Ubuntu: sudo apt update && sudo apt install -y ffmpeg
  ```

- **Hugging Face Auth / pyannote Zugriff**
  ```bash
  export HF_TOKEN="<your_token>"
  ```
  Falls `HF_TOKEN` nicht gesetzt: Der Code versucht, den Token automatisch zu laden. Für private Modelle ist ein Token meist erforderlich.

- **Modelle werden beim ersten Lauf heruntergeladen**
  - Erstaufruf kann mehrere Gigabyte Daten und Zeit benötigen.
  - Zum Testen: kleinere Modelle verwenden oder `--model <smaller-model>` angeben.

- **Out-of-memory / lange Laufzeit**
  - Pipeline berücksichtigt bereits Chunking für lange Audiodateien (>15 min ASR, >10 min Diarisierung).
  - Bei Speicherproblemen: `--num-speakers 1` reduzieren oder CPU-Optimierung verwenden.

- **macOS MPS / Torch-Fehler**
  - Wenn MPS Fehler verursacht, wird automatisch auf CPU zurückgewechselt.
  - Optional: `torch` für macOS separat installieren oder `PYTORCH_ENABLE_MPS_FALLBACK=1` setzen.

- **scripts/setup.sh: Permission denied**
  ```bash
  chmod +x scripts/setup.sh
  ./scripts/setup.sh
  ```

- **Venv nicht aktiv / falsche Python-Version**
  ```bash
  source .venv/bin/activate
  python --version  # sollte 3.10+ sein
  ```

- **Debugging & Logs**
  - Unbuffered output: `python -u main_pipline.py ...`
  - Vollständiger Output in Datei: `python main_pipline.py ... 2>&1 | tee output.log`

## Status

Das Projekt ist aktuell so aufgebaut, dass es direkt für einen Git-Upload vorbereitet werden kann: Quellcode, Dokumentation und Ignore-Regeln sind getrennt, und nur die eigentliche Implementierung soll versioniert werden.