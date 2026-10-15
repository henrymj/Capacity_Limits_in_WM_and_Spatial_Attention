#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Initialize conda for non-interactive shells, then activate the target environment.
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "conda not found on PATH" >&2
    exit 1
fi

conda activate pymc_env

for prior_sd in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50; do
    python "${SCRIPT_DIR}/6x2_fit_models_for_comparison.py" --prior_sd "${prior_sd}"
done
