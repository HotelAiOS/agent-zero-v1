#!/usr/bin/env fish
# Agent Zero V2.0 - Secure Monitoring Dashboard

set_color green; echo '📊 Starting Agent Zero V2.0 Secure Monitoring'; set_color normal
echo '🌐 Dashboard: http://localhost:8002'
echo '🛑 Press Ctrl+C to stop'
echo

source venv/bin/activate.fish 2>/dev/null; or set -gx PATH venv/bin $PATH

python3 realtime_monitor_json_fixed.py

