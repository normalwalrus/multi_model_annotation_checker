"""
runner.py — Shared engine for the per-model transcription scripts.

Each `code/<model>.py` (whisper / qwen / gemma / parakeet) is a thin wrapper
that hands this module a callable building its evaluator. This module:

  1. reads config.yml (audio folder, output folder, models, language),
  2. skips itself if the model is disabled in config,
  3. transcribes every .wav in the audio folder (full file, no segmentation),
  4. merges the predictions into a single NeMo-format manifest, one JSON object
     per line, keyed by audio_filepath, adding a `pred_text_<model>` field.

Resulting <output_folder>/manifest.json (NeMo JSONL):

  {"audio_filepath": "data/MS/000002.wav", "duration": 660.54, "text": "", "pred_text_whisper": "...", "pred_text_qwen": "...", "pred_text_gemma": "...", "pred_text_parakeet": "..."}
  ...
"""

import json
import os
import sys
from pathlib import Path
from typing import Callable

# Make the model packages (whisper_hf, qwen_model, gemma, parakeet, utils)
# importable regardless of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Repo root (= /asr-eval in the containers); config.yml lives here by default.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yml"


# ── Config loading ──────────────────────────────────────────────────────────
def _coerce(value: str):
    """Turn a YAML scalar string into bool / None / number / str."""
    v = value.strip().strip('"').strip("'")
    low = v.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~", ""):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _fallback_parse(text: str) -> dict:
    """
    Minimal YAML reader for this project's flat config with a single nested
    `models:` block. Used only when PyYAML is unavailable.
    """
    config: dict = {}
    current_block = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indented = line[0] in (" ", "\t")
        key, _, val = line.strip().partition(":")
        key = key.strip()
        if indented and current_block is not None:
            config[current_block][key] = _coerce(val)
        else:
            if val.strip() == "":
                current_block = key
                config[key] = {}
            else:
                current_block = None
                config[key] = _coerce(val)
    return config


def load_config(config_path: Path) -> dict:
    text = Path(config_path).read_text(encoding="utf-8")
    try:
        import yaml  # noqa: WPS433 (optional dependency)
        return yaml.safe_load(text)
    except ImportError:
        return _fallback_parse(text)


# ── NeMo manifest I/O (JSONL) ───────────────────────────────────────────────
def load_manifest(manifest_path: Path) -> dict[str, dict]:
    """Read a NeMo JSONL manifest into a {audio_filepath: entry} map."""
    if not manifest_path.exists():
        return {}
    entries: dict[str, dict] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "audio_filepath" in obj:
                entries[obj["audio_filepath"]] = obj
    return entries


def save_manifest(manifest_path: Path, entries: dict[str, dict]) -> None:
    """Write entries as a NeMo JSONL manifest, sorted by audio_filepath."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for key in sorted(entries):
            f.write(json.dumps(entries[key], ensure_ascii=False) + "\n")


# ── Main entry used by each <model>.py ──────────────────────────────────────
def run(model_name: str, build_evaluator: Callable, config_path: Path | None = None) -> None:
    """
    Args:
        model_name:      e.g. "whisper" — gates on config and names the output
                         field `pred_text_<model_name>`.
        build_evaluator: callable(language) -> evaluator exposing
                         `.infer_file(path) -> str`.
        config_path:     optional override; defaults to <repo>/config.yml or
                         $TRANSCRIBE_CONFIG.
    """
    config_path = Path(config_path or os.environ.get("TRANSCRIBE_CONFIG", DEFAULT_CONFIG))
    if not config_path.exists():
        sys.exit(f"[FATAL] Config not found: {config_path}")

    cfg = load_config(config_path)
    models_cfg = cfg.get("models", {}) or {}

    if not models_cfg.get(model_name, False):
        print(f"[{model_name}] disabled in {config_path.name} — skipping.")
        return

    audio_dir = Path(cfg["audio_folder"])
    output_folder = Path(cfg.get("output_folder", "outputs"))
    language = cfg.get("language") or None

    if not audio_dir.is_dir():
        sys.exit(f"[FATAL] Audio folder not found: {audio_dir}")

    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        sys.exit(f"[FATAL] No .wav files found in {audio_dir}")

    import soundfile as sf  # lazy: keeps this module importable without soundfile

    manifest_path = output_folder / "manifest.json"
    entries = load_manifest(manifest_path)
    field = f"pred_text_{model_name}"

    print(f"[{model_name}] language={language}  files={len(wav_files)}  "
          f"manifest={manifest_path}")
    print(f"[{model_name}] loading model...")
    evaluator = build_evaluator(language)
    print(f"[{model_name}] model ready. Starting inference...\n")

    for idx, wav_path in enumerate(wav_files, 1):
        key = str(wav_path)
        try:
            text = evaluator.infer_file(key)
            if isinstance(text, dict):
                text = text.get("text", "")
            text = (text or "").strip()
        except Exception as e:
            print(f"  [ERROR] {wav_path.name}: inference failed: {e}")
            text = ""

        try:
            duration = round(float(sf.info(key).duration), 2)
        except Exception:
            duration = None

        entry = entries.setdefault(key, {"audio_filepath": key, "duration": duration, "text": ""})
        if duration is not None:
            entry["duration"] = duration
        entry[field] = text
        entries[key] = entry

        # Persist after every file so a crash mid-run keeps completed work.
        save_manifest(manifest_path, entries)

        preview = text[:60] + ("…" if len(text) > 60 else "")
        print(f"  [{idx:>4}/{len(wav_files)}] {wav_path.name}  ->  {preview!r}")

    print(f"\n[{model_name}] Done. Wrote {field} for {len(wav_files)} files "
          f"-> {manifest_path}")
