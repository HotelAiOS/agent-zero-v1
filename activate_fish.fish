#!/usr/bin/env fish
# Fish shell activation for Agent Zero V2.0

set -gx VIRTUAL_ENV (pwd)/venv
set -gx PATH $VIRTUAL_ENV/bin $PATH
set -gx PYTHONPATH (pwd):$PYTHONPATH

echo "🐍 Agent Zero V2.0 virtual environment activated (Fish)"
echo "Python: $(which python)"
echo "Pip: $(which pip)"
python --version
