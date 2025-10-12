#!/bin/bash
CLI_DIR="$HOME/projects/agent-zero-v1/cli"
cd "$CLI_DIR"
source venv/bin/activate
python a0.py "$@"
