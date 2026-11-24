#!/usr/bin/env bash
# Jednorazowo, NA WĘŹLE LOGOWANIA (ma dostęp do pip). Węzły obliczeniowe tylko aktywują gotowe venv.
# (Czy węzły obliczeniowe mają internet - dokumentacja Edena nie mówi, więc instalujemy z góry.)
set -euo pipefail
python3 -m venv "$HOME/venvs/paper-soccer"
source "$HOME/venvs/paper-soccer/bin/activate"
pip install --upgrade pip
pip install -e ".[dev]"
python -c "import paper_soccer, numpy; print('OK, numpy', numpy.__version__)"
