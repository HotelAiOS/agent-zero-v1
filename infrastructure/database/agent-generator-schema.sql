-- Agent Generator Schema - przechowywanie custom agentów

CREATE TABLE IF NOT EXISTS custom_agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    system_prompt TEXT NOT NULL,
    capabilities TEXT[] DEFAULT '{}',
    tools TEXT[] DEFAULT '{}',
    model_preference VARCHAR(50) DEFAULT 'auto',
    temperature FLOAT DEFAULT 0.7,
    max_tokens INTEGER DEFAULT 2000,
    created_by VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_custom_agents_name ON custom_agents(name);
CREATE INDEX idx_custom_agents_active ON custom_agents(is_active);
CREATE INDEX idx_custom_agents_created_by ON custom_agents(created_by);

-- Agent execution history
CREATE TABLE IF NOT EXISTS agent_executions (
    id SERIAL PRIMARY KEY,
    agent_id UUID REFERENCES custom_agents(id) ON DELETE CASCADE,
    task_description TEXT,
    input_data JSONB DEFAULT '{}',
    output_data JSONB,
    tokens_used INTEGER,
    execution_time_ms FLOAT,
    status VARCHAR(50),
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_agent_executions_agent_id ON agent_executions(agent_id);
CREATE INDEX idx_agent_executions_status ON agent_executions(status);
CREATE INDEX idx_agent_executions_created_at ON agent_executions(created_at DESC);

-- Trigger dla updated_at
CREATE TRIGGER update_custom_agents_updated_at
    BEFORE UPDATE ON custom_agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Przykładowe custom agenty
INSERT INTO custom_agents (name, description, system_prompt, capabilities) VALUES
    ('TranslatorAgent', 
     'Expert translator supporting 50+ languages',
     'You are an expert translator with deep knowledge of linguistics and cultural nuances. Translate text accurately while preserving meaning, tone, and context.',
     ARRAY['translation', 'localization']),
    ('DataAnalystAgent',
     'Data analysis and visualization expert',
     'You are a data analyst expert. Analyze data, provide statistical insights, identify trends, and suggest visualizations. Use clear explanations for non-technical audiences.',
     ARRAY['data_analysis', 'statistics', 'visualization']),
    ('SecurityAuditorAgent',
     'Security code review specialist',
     'You are a cybersecurity expert specializing in code auditing. Review code for security vulnerabilities, suggest fixes, and explain security best practices.',
     ARRAY['security', 'code_review', 'auditing'])
ON CONFLICT (name) DO NOTHING;
