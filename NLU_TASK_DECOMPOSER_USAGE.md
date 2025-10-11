# NLU Task Decomposer Usage Guide

## Overview

The NLU Task Decomposer enhances Agent Zero V1 with Natural Language Understanding capabilities for intelligent task breakdown and analysis.

## Features

- 🧠 **AI-Powered Analysis**: Uses LLM for intent classification and task decomposition
- 🎯 **Context-Aware**: Considers technology stack and project type
- 📊 **Risk Assessment**: Identifies potential risk factors
- 🔗 **Dependency Detection**: Analyzes task dependencies
- 📈 **Complexity Scoring**: Estimates task complexity

## Quick Start

### 1. Basic Task Analysis

```bash
python -m cli.enhanced_task_commands task analyze "Create user authentication system"
```

### 2. With Technology Context

```bash
python -m cli.enhanced_task_commands task analyze "Build e-commerce API" \
    --tech-stack FastAPI --tech-stack PostgreSQL --tech-stack Docker \
    --project-type api_service
```

### 3. Export for Project Management

```bash
python -m cli.enhanced_task_commands task export "Develop mobile app backend" \
    --format markdown --tech-stack FastAPI --tech-stack Neo4j
```

## Programmatic Usage

```python
import asyncio
from shared.orchestration.nlp_enhanced_task_decomposer import NLUTaskDecomposer, DomainContext

async def analyze_task():
    decomposer = NLUTaskDecomposer()
    
    context = DomainContext(
        tech_stack=["FastAPI", "React", "PostgreSQL"],
        project_type="fullstack_web_app",
        current_phase="development"
    )
    
    result = await decomposer.enhanced_decompose(
        "Create user management system with role-based access control",
        context
    )
    
    print(f"Intent: {result.main_intent.primary_intent}")
    print(f"Subtasks: {len(result.subtasks)}")
    for task in result.subtasks:
        print(f"  - {task.title} ({task.estimated_hours}h)")

# Run
asyncio.run(analyze_task())
```

## Configuration

### Technology Stack Options

- **FastAPI**: Python async web framework
- **React**: JavaScript frontend framework  
- **PostgreSQL**: Relational database
- **Neo4j**: Graph database
- **Docker**: Containerization
- **Redis**: In-memory data store

### Project Types

- **fullstack_web_app**: Complete web application
- **api_service**: Backend API service
- **mobile_app**: Mobile application
- **data_pipeline**: Data processing pipeline
- **ml_project**: Machine learning project

## Advanced Features

### Custom Domain Knowledge

```python
from shared.orchestration.nlp_enhanced_task_decomposer import TechStackImplications

# Extend technology knowledge base
TechStackImplications.TECH_IMPLICATIONS["MyTech"] = {
    "implies": ["dependency1", "dependency2"],
    "common_patterns": ["pattern1", "pattern2"],
    "typical_tasks": ["task1", "task2"],
    "complexity_multiplier": 1.3
}
```

### Risk Factor Analysis

The system automatically identifies:
- High complexity technologies
- Task volume risks
- Integration complexity
- Dependency conflicts

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure setup script completed successfully
2. **spaCy Model Missing**: Run `python -m spacy download en_core_web_sm`
3. **LLM Connection**: Check Ollama is running and accessible

### Fallback Mode

If AI components fail, the system falls back to:
- Rule-based entity extraction
- Pattern-based task generation
- Simple dependency detection

## Integration with Existing Systems

The NLU Task Decomposer integrates with:
- Existing `TaskDecomposer` class
- Agent Zero CLI system
- Project orchestration components
- Task scheduling system

## Performance Tips

1. **Cache Results**: Store decomposition results for similar tasks
2. **Batch Processing**: Process multiple tasks together
3. **Context Reuse**: Reuse domain contexts for similar projects
4. **Model Selection**: Use appropriate LLM size for your needs

## Next Steps

1. Explore advanced configuration options
2. Integrate with project management tools
3. Customize for your specific domain
4. Extend with additional AI capabilities

For more information, see the Agent Zero V1 documentation.
