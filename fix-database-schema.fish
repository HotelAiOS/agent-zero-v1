#!/usr/bin/env fish
# 🔧 Agent Zero V2.0 - Database Schema Quick Fix
# 🗄️ Naprawia konflikt schematów bazy danych

echo "🔧 Agent Zero V2.0 - Database Schema Fix"
echo "📅 " (date)

# Remove conflicting database
echo "🗄️ Removing conflicting database..."
rm -f agent_zero.db* 2>/dev/null

# Create fresh database with correct schema
echo "🆕 Creating fresh database..."
python3 -c "
import sqlite3
from datetime import datetime

conn = sqlite3.connect('agent_zero.db')

# Create correct schema matching agent_zero_standalone.py
conn.executescript('''
-- Agents table (compatible with agent_zero_standalone.py)
CREATE TABLE agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    email TEXT,
    skills TEXT,
    status TEXT DEFAULT 'available',
    cost_per_hour REAL DEFAULT 100.0,
    timezone TEXT DEFAULT 'UTC',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Projects table  
CREATE TABLE projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'planning',
    budget REAL,
    deadline DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Team history table
CREATE TABLE team_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    team_composition TEXT NOT NULL,
    outcome_success REAL NOT NULL,
    budget_delta REAL DEFAULT 0.0,
    timeline_delta REAL DEFAULT 0.0,
    quality_score REAL DEFAULT 0.0,
    team_satisfaction REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analytics reports table
CREATE TABLE analytics_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    report_data TEXT NOT NULL,
    format TEXT NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP DEFAULT (datetime(\"now\", \"+7 days\"))
);

-- Insert sample data with correct columns
INSERT INTO agents (agent_id, name, email, skills, status, cost_per_hour, timezone) VALUES
(\"dev_001\", \"Alice Developer\", \"alice@agent-zero.ai\", \"{\\\"python\\\": 0.9, \\\"fastapi\\\": 0.8, \\\"sql\\\": 0.7, \\\"ai\\\": 0.6}\", \"available\", 120.0, \"UTC+1\"),
(\"dev_002\", \"Bob Designer\", \"bob@agent-zero.ai\", \"{\\\"figma\\\": 0.9, \\\"ui_ux\\\": 0.95, \\\"html_css\\\": 0.8, \\\"prototyping\\\": 0.85}\", \"available\", 100.0, \"UTC+1\"),
(\"dev_003\", \"Charlie DevOps\", \"charlie@agent-zero.ai\", \"{\\\"docker\\\": 0.9, \\\"kubernetes\\\": 0.8, \\\"aws\\\": 0.85, \\\"ci_cd\\\": 0.75}\", \"available\", 130.0, \"UTC\"),
(\"dev_004\", \"Diana PM\", \"diana@agent-zero.ai\", \"{\\\"project_management\\\": 0.95, \\\"agile\\\": 0.9, \\\"stakeholder_mgmt\\\": 0.85, \\\"budgeting\\\": 0.8}\", \"available\", 110.0, \"UTC+2\"),
(\"dev_005\", \"Eve QA\", \"eve@agent-zero.ai\", \"{\\\"testing\\\": 0.9, \\\"automation\\\": 0.8, \\\"selenium\\\": 0.75, \\\"performance\\\": 0.7}\", \"available\", 90.0, \"UTC\");

INSERT INTO projects (project_id, name, description, status, budget, deadline) VALUES
(\"proj_001\", \"E-commerce Platform V2\", \"Next-gen e-commerce platform with AI recommendations\", \"active\", 75000.0, \"2024-12-31\"),
(\"proj_002\", \"Mobile Banking App\", \"Secure mobile banking with biometric auth\", \"planning\", 95000.0, \"2024-11-30\"),
(\"proj_003\", \"AI Analytics Dashboard\", \"Real-time analytics with ML insights\", \"active\", 60000.0, \"2024-10-31\"),
(\"proj_004\", \"IoT Smart Home Hub\", \"Centralized IoT device management\", \"planning\", 45000.0, \"2025-01-15\");

INSERT INTO team_history (project_id, team_composition, outcome_success, budget_delta, timeline_delta, quality_score, team_satisfaction) VALUES
(\"proj_001\", \"[\\\"dev_001\\\", \\\"dev_002\\\", \\\"dev_003\\\"]\", 0.85, -0.05, 0.1, 0.9, 0.8),
(\"proj_002\", \"[\\\"dev_001\\\", \\\"dev_004\\\", \\\"dev_005\\\"]\", 0.75, 0.15, -0.05, 0.8, 0.7),
(\"proj_003\", \"[\\\"dev_002\\\", \\\"dev_003\\\", \\\"dev_004\\\"]\", 0.92, -0.1, -0.08, 0.95, 0.9);
''')

conn.commit()
conn.close()
print('✅ Database created with correct schema and full sample data')
"

# Verify database
echo "🧪 Verifying database..."
python3 -c "
import sqlite3

conn = sqlite3.connect('agent_zero.db')
cursor = conn.cursor()

# Check schema
cursor.execute('PRAGMA table_info(agents)')
columns = [row[1] for row in cursor.fetchall()]
print(f'✅ Agents columns: {columns}')

# Check data
cursor.execute('SELECT COUNT(*) FROM agents')
agent_count = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM projects')  
project_count = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM team_history')
history_count = cursor.fetchone()[0]

print(f'✅ Data loaded: {agent_count} agents, {project_count} projects, {history_count} history records')

conn.close()
"

echo ""
echo "🎉 Database Schema Fix - COMPLETE!"
echo "✅ Schema compatible with agent_zero_standalone.py"
echo "✅ Full sample data loaded (5 agents, 4 projects, 3 history)"
echo ""
echo "🚀 Now run the full version:"
echo "   source venv/bin/activate.fish"
echo "   python3 agent_zero_standalone.py"