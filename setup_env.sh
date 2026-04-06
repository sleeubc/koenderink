#!/bin/zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
USER_NAME="${USER:-$(whoami)}"
VENV_DIR="${PROJECT_DIR}/.venv-${USER_NAME}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Using environment: ${VENV_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_DIR}/requirements.txt"

echo
echo "Environment ready."
echo "Activate with:"
echo "  source ${VENV_DIR}/bin/activate"

