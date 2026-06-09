# ST Annotation Checking — Multi-Model ASR Transcription

Transcribe a folder of `.wav` files with up to **four ASR models** —
**Whisper**, **Qwen3-ASR**, **Gemma**, and **NVIDIA Parakeet** — and collect
every model's prediction into a single **NeMo-format** manifest.

No input manifest is required: point the tool at a folder of audio, choose which
models to run in `config.yml`, and it produces, for **every** `.wav`, the
transcription from each enabled model.

`outputs/manifest.json` (NeMo JSONL — one JSON object per line):

```json
{"audio_filepath": "data/MS/000002.wav", "duration": 660.54, "text": "", "pred_text_whisper": "...", "pred_text_qwen": "...", "pred_text_gemma": "...", "pred_text_parakeet": "..."}
```

---

## How it works

Each model lives in **its own Docker image** with its own (conflicting)
dependencies, so a single Python process can't load all of them. The design is:

- **`config.yml`** — the single source of settings: audio folder, output folder,
  which models to run, and language.
- **`code/<model>.py`** — one thin runner per model (`whisper.py`, `qwen.py`,
  `gemma.py`, `parakeet.py`). Each reads `config.yml`, skips itself if disabled,
  transcribes every `.wav`, and merges its prediction into the shared NeMo
  manifest. The shared logic lives in **`code/runner.py`**.
- **`run_all.sh`** — the single command. It reads `config.yml`, then runs each
  **enabled** model in its own container, in turn (they share the one GPU), each
  pass adding its own `pred_text_<model>` field to the same manifest.

```
run_all.sh  ──reads──►  config.yml (which models? where?)
   │
   ├─ docker compose run whisper-hf → python code/whisper.py   ─┐
   ├─ docker compose run qwen       → python code/qwen.py      ─┤
   ├─ docker compose run gemma      → python code/gemma.py     ─┼─► outputs/manifest.json
   └─ docker compose run parakeet   → python code/parakeet.py  ─┘   (NeMo JSONL)
```

| Model    | Container service | Model source                         | Language hint |
|----------|-------------------|--------------------------------------|---------------|
| Whisper  | `whisper-hf`      | `pretrained_models/whisper-large-v3` | yes           |
| Qwen     | `qwen`            | `Qwen/Qwen3-ASR-1.7B`                | yes           |
| Gemma    | `gemma`           | `google/gemma-4-E2B-it`             | yes           |
| Parakeet | `parakeet`        | `nvidia/parakeet-tdt-1.1b`          | no (ignored)  |

`runner.py` saves the manifest after **every file**, so a crash mid-run keeps
all completed work, and re-running a model overwrites only its own field.

---

## Repository layout

```
.
├── config.yml                # ← all settings: audio/output/models/language
├── run_all.sh                # ← one command: run enabled models, merge manifest
├── docker-compose.yml        # whisper-hf / qwen / gemma / parakeet (GPU)
├── .env                      # host paths for mounted volumes (edit this)
├── code/
│   ├── runner.py             # shared engine (config -> transcribe -> NeMo manifest)
│   ├── whisper.py            # ┐
│   ├── qwen.py               # ├ per-model runners (thin wrappers over runner.py)
│   ├── gemma.py              # │
│   ├── parakeet.py           # ┘
│   ├── whisper_hf/           # Whisper model package + Dockerfile
│   ├── qwen_model/           # Qwen model package + Dockerfile
│   ├── gemma/                # Gemma model package + Dockerfile
│   ├── parakeet/             # Parakeet model package + Dockerfile
│   └── utils/                # audio / file / language helpers
├── pretrained_models/        # local model weights (e.g. whisper-large-v3)
├── data/                     # mounted dataset (your .wav folders live here)
└── outputs/                  # manifest.json
```

---

## Prerequisites

- Docker + Docker Compose
- NVIDIA GPU + the NVIDIA Container Toolkit (each service requests 1 GPU)
- Whisper weights downloaded to `pretrained_models/whisper-large-v3`
  (see `code/whisper_hf/README.md` for the required files)

### 1. Configure `.env`

Point these at the host directories mounted into the containers:

```env
PRETRAINED_MODELS_PATH=/abs/path/to/pretrained_models
BENCHMARK_DATASET_PATH=/abs/path/to/your/data     # mounted at /asr-eval/data
LORA_CHECKPOINT_PATH=/abs/path/to/checkpoints
```

> Your audio folders must live under `BENCHMARK_DATASET_PATH`, because that
> directory is mounted into every container at `/asr-eval/data`.

### 2. Build the images

```bash
docker compose build                 # whisper-hf, qwen, gemma, parakeet
# or individually, e.g.:
docker compose build parakeet
```

---

## Configure the run — `config.yml`

```yaml
# Folder of .wav files to transcribe (path as seen inside the container).
# Host BENCHMARK_DATASET_PATH is mounted at /asr-eval/data, so e.g. data/MS.
audio_folder: data/MS

# Where the NeMo manifest is written (-> <output_folder>/manifest.json).
output_folder: outputs

# Language hint for models that accept one (ms, malay, en, ...).
# Leave blank / null for auto-detect. (Parakeet ignores this.)
language: ms

# Which models to run. Set to false to skip a model entirely.
models:
  whisper: true
  qwen: true
  gemma: false      # ← example: skip Gemma
  parakeet: true
```

---

## Run

```bash
./run_all.sh                 # uses config.yml
./run_all.sh my_config.yml   # or a custom config
```

`run_all.sh` launches only the models set to `true`. Each enabled model runs in
its container and appends `pred_text_<model>` to `outputs/manifest.json`.

### Run a single model

```bash
docker compose run --rm qwen -c "\
  cd /asr-eval && PYTHONPATH=/asr-eval/code python code/qwen.py"
```

(A model disabled in `config.yml` will print a skip message and exit.)

---

## Output

A single **NeMo-format** manifest (JSONL — one JSON object per line) at
`<output_folder>/manifest.json`, one line per audio file:

```json
{"audio_filepath": "data/MS/000002.wav", "duration": 660.54, "text": "", "pred_text_whisper": "...", "pred_text_qwen": "...", "pred_text_gemma": "...", "pred_text_parakeet": "..."}
```

- `audio_filepath`, `duration`, and `text` are standard NeMo fields (`text` is
  left empty — there is no ground-truth reference).
- Each model contributes a `pred_text_<model>` prediction field. Only enabled
  models appear.
- If a model fails on a file, its field is set to `""` and processing continues.

---

## Notes & limitations

- **Full-file transcription:** with no input manifest there is no
  segmentation — each whole `.wav` is transcribed in one pass. Qwen
  (`max_new_tokens=256`) and Gemma (`1024`) will **truncate** long recordings.
  For long audio, add VAD/chunking before inference.
- **Sequential, not parallel:** each service reserves the single GPU, so models
  run one after another. This also avoids manifest write races.
- **Idempotent:** re-running a model overwrites only its own `pred_text_<model>`
  field; other models' predictions in the manifest are preserved.
- **Config parsing:** `runner.py` uses PyYAML if present, with a built-in
  fallback parser, so it works even in an image without PyYAML installed.

---

## Adding another model

1. Create `code/<name>/` with `src/transcribe.py` exposing an evaluator that has
   `infer_file(path) -> str` (add an empty `code/<name>/__init__.py` if the
   runner script will share the package's name).
2. Add `code/<name>.py` (copy an existing one; point `build()` at your
   evaluator).
3. Add a `<name>` service to `docker-compose.yml` and a `[<name>]=<service>`
   entry in `run_all.sh`.
4. Add `<name>: true` under `models:` in `config.yml`.
