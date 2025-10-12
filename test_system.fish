#!/usr/bin/env fish
# Agent Zero V2.0 - Testing Script

echo "🧪 Testing Agent Zero V2.0"
source venv/bin/activate.fish

echo "🗄️ Testing Database..."
python3 -c "
import sqlite3
try:
    conn = sqlite3.connect('agent_zero.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM sqlite_master WHERE type=\"table\"')
    count = cursor.fetchone()[0]
    print(f'✅ Database: {count} tables')
    conn.close()
except Exception as e:
    print(f'❌ Database error: {e}')
"

echo "🤖 Testing Core Components..."
python3 -c "
try:
    from agent_zero_missing_features_production_implementation import ExperienceRepository, ReportExporter
    repo = ExperienceRepository()
    exporter = ReportExporter()
    print('✅ Core components: OK')
except Exception as e:
    print(f'❌ Component error: {e}')
"

echo "✅ Testing completed!"
