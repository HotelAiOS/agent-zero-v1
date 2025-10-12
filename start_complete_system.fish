#!/usr/bin/env fish
# Agent Zero V2.0 - Complete Secure System

set_color magenta; echo '🏰 Starting Complete Agent Zero V2.0 Secure System'; set_color normal
echo '🔐 Secure API: http://localhost:8003'
echo '📊 Monitoring: http://localhost:8002'  
echo '🛑 Press Ctrl+C to stop all services'
echo

source venv/bin/activate.fish 2>/dev/null; or set -gx PATH venv/bin $PATH

# Start monitoring in background
python3 realtime_monitor_json_fixed.py &
set monitor_pid $last_pid

# Start secure API
python3 -c "
import uvicorn
from security_integration_layer import create_secure_api, HAS_SECURE_API

if HAS_SECURE_API:
    app = create_secure_api()
    print('🚀 Starting complete secure system...')
    uvicorn.run(app, host='0.0.0.0', port=8003, log_level='info')
else:
    print('❌ Running basic monitoring only')
    import asyncio
    asyncio.get_event_loop().run_forever()
"

# Cleanup on exit
kill $monitor_pid 2>/dev/null

