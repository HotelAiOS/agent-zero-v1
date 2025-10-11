// Agent Zero Knowledge Graph Schema
// Nodes: Agent, Task, CodePattern, Experience, Project

// Create constraints and indexes
CREATE CONSTRAINT agent_id_unique IF NOT EXISTS FOR (a:Agent) REQUIRE a.agent_id IS UNIQUE;
CREATE CONSTRAINT task_id_unique IF NOT EXISTS FOR (t:Task) REQUIRE t.task_id IS UNIQUE;
CREATE CONSTRAINT project_id_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.project_id IS UNIQUE;
CREATE INDEX agent_type_idx IF NOT EXISTS FOR (a:Agent) ON (a.agent_type);
CREATE INDEX task_status_idx IF NOT EXISTS FOR (t:Task) ON (t.status);
CREATE INDEX experience_timestamp_idx IF NOT EXISTS FOR (e:Experience) ON (e.timestamp);

// Sample schema documentation
// :Agent {agent_id, agent_type, created_at, capabilities[]}
// :Task {task_id, description, status, complexity, created_at, completed_at}
// :CodePattern {pattern_id, language, pattern_type, code_snippet, usage_count}
// :Experience {experience_id, context, outcome, success, timestamp}
// :Project {project_id, name, description, status, created_at}

// Relationships:
// (Agent)-[:EXECUTED]->(Task)
// (Agent)-[:LEARNED]->(Experience)
// (Agent)-[:USED_PATTERN]->(CodePattern)
// (Task)-[:BELONGS_TO]->(Project)
// (Task)-[:DEPENDS_ON]->(Task)
// (Agent)-[:COLLABORATED_WITH]->(Agent)
