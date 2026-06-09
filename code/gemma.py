"""gemma.py — run the Gemma model over the configured audio folder.

Reads config.yml (audio folder, output folder, models, language) and writes
its predictions into the shared NeMo manifest. Run inside the gemma container:
    python code/gemma.py
"""

from runner import run


def build(language):
    from gemma.src.transcribe import GemmaEvaluator
    return GemmaEvaluator(
        language=language,
        model_dir="google/gemma-4-E2B-it",
    )


if __name__ == "__main__":
    run("gemma", build)
