"""whisper.py — run the Whisper model over the configured audio folder.

Reads config.yml (audio folder, output folder, models, language) and writes
its predictions into the shared NeMo manifest. Run inside the whisper-hf
container:  python code/whisper.py
"""

from runner import run


def build(language):
    from whisper_hf.src.transcribe import WhisperEvaluator
    return WhisperEvaluator(
        language=language,
        model_dir="pretrained_models/whisper-large-v3",
    )


if __name__ == "__main__":
    run("whisper", build)
