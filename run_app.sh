#!/bin/zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
USER_NAME="${USER:-$(whoami)}"
VENV_DIR="${PROJECT_DIR}/.venv-${USER_NAME}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing virtual environment: ${VENV_DIR}"
  echo "Run ./setup_env.sh first."
  exit 1
fi

source "${VENV_DIR}/bin/activate"
exec streamlit run "${PROJECT_DIR}/app.py"

