#!/usr/bin/env bash
#
# run_all.sh — Transcribe every .wav in the configured folder with the models
#              enabled in config.yml (whisper / qwen / gemma / parakeet) and
#              merge them into one NeMo manifest.
#
# All settings come from config.yml (audio_folder, output_folder, language,
# and which models to run). No command-line arguments needed.
#
# Usage:
#   ./run_all.sh [config.yml]
#
# Each enabled model runs in its own container, in turn (they share the single
# GPU), each pass adding its own `pred_text_<model>` field to the same
# <output_folder>/manifest.json.
#
set -euo pipefail

CONFIG="${1:-config.yml}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[FATAL] Config not found: ${CONFIG}" >&2
  exit 1
fi

# model name -> docker compose service name
declare -A SERVICE=(
  [whisper]="whisper-hf"
  [qwen]="qwen"
  [gemma]="gemma"
  [parakeet]="parakeet"
)

# Extract the models set to `true` under the `models:` block of config.yml.
mapfile -t ENABLED < <(awk '
  /^[A-Za-z0-9_]/      { in_models = ($1 == "models:") }
  in_models && /^[[:space:]]+[A-Za-z0-9_]+[[:space:]]*:/ {
    line = $0; sub(/#.*/, "", line); gsub(/[[:space:]]/, "", line)
    split(line, kv, ":")
    if (kv[2] == "true") print kv[1]
  }
' "${CONFIG}")

if [[ ${#ENABLED[@]} -eq 0 ]]; then
  echo "[FATAL] No models enabled in ${CONFIG} (models: <name>: true)." >&2
  exit 1
fi

echo "Enabled models: ${ENABLED[*]}"

for model in "${ENABLED[@]}"; do
  service="${SERVICE[$model]:-}"
  if [[ -z "${service}" ]]; then
    echo ">> Skipping unknown model '${model}' (no container mapping)."
    continue
  fi

  echo "==================================================================="
  echo ">> Running '${model}' in container '${service}'"
  echo "==================================================================="

  # ENTRYPOINT of each image is bash, so we pass a `-c` command string.
  # PYTHONPATH=/asr-eval/code makes the model packages importable; cwd
  # /asr-eval keeps config.yml, data/ and outputs/ paths resolvable.
  docker compose run --rm "${service}" -c "\
    cd /asr-eval && \
    PYTHONPATH=/asr-eval/code python code/${model}.py"
done

echo
echo "Done. Combined NeMo manifest -> see output_folder in ${CONFIG} (outputs/manifest.json)"
