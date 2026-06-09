"""qwen.py — run the Qwen3-ASR model over the configured audio folder.

Reads config.yml (audio folder, output folder, models, language) and writes
its predictions into the shared NeMo manifest. Run inside the qwen container:
    python code/qwen.py
"""

from runner import run


def build(language):
    from qwen_model.src.transcribe import QwenEvaluator
    return QwenEvaluator(
        language=language,
        model_dir="Qwen/Qwen3-ASR-1.7B",
    )


if __name__ == "__main__":
    run("qwen", build)
