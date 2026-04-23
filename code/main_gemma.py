from gemma.src.transcribe import GemmaEvaluator
from utils.audio_utils import get_segment_of_audio_wav, get_audio_wav_and_sample_rate
from utils.file_utils import read_json_file
from utils.language_utils import filter_string
from jiwer import wer as compute_wer
import tempfile
import soundfile as sf

import json
import os
import numpy as np
from pathlib import Path

import os

# ── Config ────────────────────────────────────────────────────────────────────
LANGUAGE                = "ms"
PATH_TO_MANIFEST_FOLDER = "data/MALT_ST_ANNOTATED_RESULT/ms_whisper"
PATH_TO_AUDIO_FOLDER    = "data/to_pass_to_st/passed_30h_st_110326/MS"
OUTPUT_FOLDER           = "outputs/gemma"
 
MODEL = GemmaEvaluator(
    language=LANGUAGE,
    model_dir="google/gemma-4-E2B-it",
)
# ─────────────────────────────────────────────────────────────────────────────

def safe_wer(reference: str, hypothesis: str, language: str) -> float | None:
    """
    Return WER as a float (0.0 – 1.0+).
    Returns None when the reference is empty (nothing to evaluate against).
    """
    ref = filter_string(
        reference,
        lang=language
    )
    hyp = filter_string(
        hypothesis,
        lang=language
    )
    if not ref:          # empty ground-truth → WER undefined
        return None
    if not hyp:          # empty hypothesis → all words deleted
        return 1.0
    return compute_wer(ref, hyp)


def process_manifest(manifest_path: Path, audio_folder: str, output_folder: str) -> None:
    manifest = read_json_file(manifest_path)

    audio_filename = manifest["filename"]
    audio_path     = os.path.join(audio_folder, audio_filename)
 
    if not os.path.exists(audio_path):
        print(f"  [WARN] Audio not found, skipping: {audio_path}")
        return
 
    audio, sr = get_audio_wav_and_sample_rate(audio_path)
 
    enriched_annotations = []
    for idx, segment in enumerate(manifest.get("annotations", [])):
        start = float(segment["start"])
        end   = float(segment["end"])
        ref   = segment.get("transcription", "")
 
        # ── Slice & infer ────────────────────────────────────────────────────
        audio_slice, sr = get_segment_of_audio_wav(
            wav=audio, 
            start_time=start,
            end_time=end,
            sr=sr
        )

        if len(audio_slice) == 0:
            print(f"  [WARN] Empty slice at segment {idx} ({start}–{end}s), skipping.")
            enriched_annotations.append({
                **segment,
                "gemma_transcription": "",
                "gemma_wer": None,
            })
            continue
 
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, audio_slice, sr)
                whisper_text = MODEL.infer(tmp.name)

            if isinstance(whisper_text, dict):
                whisper_text = whisper_text.get("text", "")
            whisper_text = (whisper_text or "").strip()

        except Exception as e:
            print(f"  [ERROR] Inference failed at segment {idx}: {e}")
            whisper_text = ""
 
        # ── WER ──────────────────────────────────────────────────────────────
        whisper_wer_value = safe_wer(
            ref, 
            whisper_text,
            language=LANGUAGE
        )
 
        enriched_annotations.append({
            **segment,
            "gemma_transcription": whisper_text,
            "gemma_wer": whisper_wer_value,
        })
 
        print(
            f"  seg {idx:03d} [{start:.2f}–{end:.2f}s]  "
            f"WER={whisper_wer_value:.3f}" if whisper_wer_value is not None
            else f"  seg {idx:03d} [{start:.2f}–{end:.2f}s]  WER=N/A (empty ref)"
        )
 
    # ── Write output manifest ─────────────────────────────────────────────────
    output_manifest = {**manifest, "annotations": enriched_annotations}
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, manifest_path.name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_manifest, f, ensure_ascii=False, indent=2)
 
    print(f"  ✓ Saved → {out_path}")

def main() -> None:
    manifest_dir = Path(PATH_TO_MANIFEST_FOLDER)
    manifest_files = sorted(manifest_dir.glob("*.json"))

    if not manifest_files:
        print(f"No manifest JSON files found in {PATH_TO_MANIFEST_FOLDER}")
        return
 
    print(f"Found {len(manifest_files)} manifest(s). Starting inference...\n")
 
    for manifest_path in manifest_files:
        print(f"Processing: {manifest_path.name}")
        process_manifest(manifest_path, PATH_TO_AUDIO_FOLDER, OUTPUT_FOLDER)
 
    print("\nDone.") 

if __name__ == "__main__":
    main()
 
