#!/usr/bin/env fish
# Agent Zero V2.0 - Secure API Server

set_color blue; echo '🔐 Starting Agent Zero V2.0 Secure API Server'; set_color normal
echo '📊 Dashboard will be available at: http://localhost:8003'
echo '🔑 Use POST /api/auth/login to authenticate'
echo '🛑 Press Ctrl+C to stop'
echo

source venv/bin/activate.fish 2>/dev/null; or set -gx PATH venv/bin $PATH

python3 -c "
import uvicorn
from security_integration_layer import create_secure_api, HAS_SECURE_API

if HAS_SECURE_API:
    app = create_secure_api()
    print('🚀 Starting secure API server...')
    uvicorn.run(app, host='0.0.0.0', port=8003, log_level='info')
else:
    print('❌ Secure API dependencies not available')
    print('💡 Run: pip install fastapi uvicorn')
"

