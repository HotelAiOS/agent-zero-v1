-- Learning System Schema - system uczenia się z historii

CREATE TABLE IF NOT EXISTS interaction_feedback (
    id SERIAL PRIMARY KEY,
    session_id UUID,
    message_id UUID,
    user_id VARCHAR(255),
    feedback_type VARCHAR(50), -- positive, negative, neutral
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_feedback_session_id ON interaction_feedback(session_id);
CREATE INDEX idx_feedback_user_id ON interaction_feedback(user_id);
CREATE INDEX idx_feedback_type ON interaction_feedback(feedback_type);

CREATE TABLE IF NOT EXISTS learning_insights (
    id SERIAL PRIMARY KEY,
    insight_type VARCHAR(100), -- popular_query, common_error, user_preference
    content JSONB NOT NULL,
    frequency INTEGER DEFAULT 1,
    confidence_score FLOAT DEFAULT 0.5,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_insights_type ON learning_insights(insight_type);
CREATE INDEX idx_insights_frequency ON learning_insights(frequency DESC);
CREATE INDEX idx_insights_confidence ON learning_insights(confidence_score DESC);

CREATE TABLE IF NOT EXISTS user_patterns (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    pattern_type VARCHAR(100), -- preferred_model, common_task, time_pattern
    pattern_data JSONB NOT NULL,
    occurrences INTEGER DEFAULT 1,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_patterns_user_id ON user_patterns(user_id);
CREATE INDEX idx_patterns_type ON user_patterns(pattern_type);

-- Funkcja do agregacji insights
CREATE OR REPLACE FUNCTION update_learning_insights()
RETURNS void AS $$
BEGIN
    -- Najpopularniejsze zapytania
    INSERT INTO learning_insights (insight_type, content, frequency)
    SELECT 
        'popular_query',
        jsonb_build_object('query', content, 'avg_tokens', AVG(tokens)),
        COUNT(*)
    FROM chat_messages
    WHERE role = 'user' 
    AND created_at > NOW() - INTERVAL '7 days'
    GROUP BY content
    HAVING COUNT(*) > 5
    ON CONFLICT DO NOTHING;
    
    -- User patterns - preferowane modele
    INSERT INTO user_patterns (user_id, pattern_type, pattern_data, occurrences)
    SELECT 
        cm.metadata->>'user_id',
        'preferred_model',
        jsonb_build_object('model', cm.metadata->>'model'),
        COUNT(*)
    FROM chat_messages cm
    WHERE cm.metadata->>'model' IS NOT NULL
    AND cm.created_at > NOW() - INTERVAL '7 days'
    GROUP BY cm.metadata->>'user_id', cm.metadata->>'model'
    HAVING COUNT(*) > 3
    ON CONFLICT (user_id, pattern_type) DO UPDATE
    SET occurrences = user_patterns.occurrences + EXCLUDED.occurrences,
        last_updated = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Funkcja do analizy user behavior
CREATE OR REPLACE FUNCTION analyze_user_behavior(p_user_id VARCHAR)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_messages', COUNT(*),
        'avg_message_length', AVG(LENGTH(content)),
        'most_active_hour', MODE() WITHIN GROUP (ORDER BY EXTRACT(HOUR FROM created_at)),
        'preferred_task_types', (
            SELECT jsonb_agg(DISTINCT metadata->>'task_type')
            FROM chat_messages
            WHERE metadata->>'user_id' = p_user_id
        ),
        'last_7_days_activity', (
            SELECT COUNT(*)
            FROM chat_messages
            WHERE metadata->>'user_id' = p_user_id
            AND created_at > NOW() - INTERVAL '7 days'
        )
    ) INTO result
    FROM chat_messages
    WHERE metadata->>'user_id' = p_user_id;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;
