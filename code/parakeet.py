"""parakeet.py — run the NVIDIA Parakeet model over the configured audio folder.

Reads config.yml (audio folder, output folder, models, language) and writes
its predictions into the shared NeMo manifest. Run inside the parakeet
container:  python code/parakeet.py

Note: Parakeet does not take a language hint; the `language` arg is ignored.
"""

from runner import run


def build(language):  # language unused — Parakeet has no language hint
    from parakeet.src.transcribe import ParakeetEvaluator
    # Matches the model pre-cached in code/parakeet/Dockerfile.
    return ParakeetEvaluator(
        model_tag="nvidia/parakeet-tdt-1.1b",
    )


if __name__ == "__main__":
    run("parakeet", build)
