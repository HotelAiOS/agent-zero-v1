#!/usr/bin/env python3
"""
Agent Zero V1 - V2.0 Demo Data Generator
Generuje przykładowe dane dla demonstracji V2.0 Intelligence Layer

Author: Developer A (Backend Architect)
Date: 10 października 2025, 18:08 CEST
Linear Issue: A0-28
"""

import sys
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import random

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def generate_demo_data():
    """Generate realistic demo data for V2.0 Intelligence Layer"""
    
    print("🎭 Generating V2.0 Demo Data...")
    
    try:
        # Import V2.0 components
        from shared.kaizen.intelligent_selector import IntelligentModelSelector, TaskType
        from shared.kaizen.success_evaluator import SuccessEvaluator, TaskResult, TaskOutputType
        from shared.kaizen.metrics_analyzer import ActiveMetricsAnalyzer
        
        # Initialize components
        selector = IntelligentModelSelector("agent_zero.db")
        evaluator = SuccessEvaluator("agent_zero.db")
        analyzer = ActiveMetricsAnalyzer("agent_zero.db")
        
        # Demo scenarios
        demo_tasks = [
            {
                "task_type": TaskType.CODE_GENERATION,
                "complexity": 1.2,
                "description": "Generate REST API endpoints",
                "expected_output": "def create_user(request):\n    return {'status': 'success'}",
                "execution_time": 2500,
                "cost": 0.012
            },
            {
                "task_type": TaskType.CODE_REVIEW,
                "complexity": 0.8,
                "description": "Review Python function",
                "expected_output": "Code looks good, consider error handling",
                "execution_time": 1800,
                "cost": 0.008
            },
            {
                "task_type": TaskType.DOCUMENTATION,
                "complexity": 1.0,
                "description": "Write API documentation",
                "expected_output": "# API Documentation\n\nThis endpoint creates...",
                "execution_time": 3200,
                "cost": 0.015
            },
            {
                "task_type": TaskType.DEBUGGING,
                "complexity": 1.5,
                "description": "Fix authentication bug",
                "expected_output": "Found issue in token validation logic",
                "execution_time": 4500,
                "cost": 0.025
            },
            {
                "task_type": TaskType.ARCHITECTURE_DESIGN,
                "complexity": 2.0,
                "description": "Design microservice architecture",
                "expected_output": "Proposed architecture with 5 microservices",
                "execution_time": 8000,
                "cost": 0.045
            },
            {
                "task_type": TaskType.BUSINESS_ANALYSIS,
                "complexity": 1.3,
                "description": "Analyze user requirements",
                "expected_output": "Users need faster response times",
                "execution_time": 5500,
                "cost": 0.032
            }
        ]
        
        print("📊 Processing demo scenarios...")
        
        # Generate data for the last 7 days
        base_date = datetime.now() - timedelta(days=7)
        
        for day in range(7):
            current_date = base_date + timedelta(days=day)
            
            # Process 2-4 tasks per day
            tasks_per_day = random.randint(2, 4)
            
            for task_idx in range(tasks_per_day):
                # Select random task scenario
                scenario = random.choice(demo_tasks)
                
                # Add some randomness
                variation = random.uniform(0.8, 1.2)
                execution_time = int(scenario["execution_time"] * variation)
                cost = scenario["cost"] * variation
                
                # Generate AI recommendation
                context = {
                    'complexity': scenario["complexity"] * variation,
                    'urgency': random.uniform(0.8, 1.2),
                    'budget': random.choice(['low', 'medium', 'high'])
                }
                
                recommendation = selector.recommend_model(scenario["task_type"], context)
                
                # Store model decision manually with backdated timestamp
                with sqlite3.connect("agent_zero.db") as conn:
                    conn.execute('''
                        INSERT INTO v2_model_decisions (
                            timestamp, task_type, recommended_model, confidence_score,
                            reasoning, alternatives, estimated_cost, estimated_latency_ms,
                            context_factors
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        current_date + timedelta(hours=task_idx*2),
                        scenario["task_type"].value,
                        recommendation.model_name,
                        recommendation.confidence_score,
                        recommendation.reasoning,
                        json.dumps(recommendation.alternatives),
                        cost,
                        execution_time,
                        json.dumps(context)
                    ))
                
                # Create task result
                task_result = TaskResult(
                    task_id=f"demo_{day}_{task_idx}",
                    task_type=scenario["task_type"].value,
                    output_type=TaskOutputType.CODE if "code" in scenario["description"].lower() else TaskOutputType.TEXT,
                    output_content=scenario["expected_output"],
                    expected_requirements=[scenario["description"]],
                    context=context,
                    execution_time_ms=execution_time,
                    cost_usd=cost,
                    model_used=recommendation.model_name,
                    human_feedback=random.randint(1, 5) if random.random() > 0.7 else None
                )
                
                # Evaluate task (this will store in database with current timestamp)
                evaluation = evaluator.evaluate_task(task_result)
                
                # Update the timestamp to match the demo day
                with sqlite3.connect("agent_zero.db") as conn:
                    conn.execute('''
                        UPDATE v2_success_evaluations 
                        SET timestamp = ?
                        WHERE task_id = ?
                    ''', (current_date + timedelta(hours=task_idx*2), task_result.task_id))
                
                # Analyze for metrics (this stores in v2_active_metrics)
                task_completion_data = {
                    'cost_usd': cost,
                    'execution_time_ms': execution_time,
                    'success': evaluation.level.value in ['SUCCESS', 'PARTIAL'],
                    'human_override': random.random() < 0.1
                }
                
                analyzer.analyze_task_completion(task_completion_data)
                
                print(f"  ✅ Day {day+1}: {scenario['task_type'].value} → {recommendation.model_name} (score: {evaluation.overall_score:.2f})")
        
        print("\n📈 Demo data generation complete!")
        
        # Show summary
        with sqlite3.connect("agent_zero.db") as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM v2_model_decisions")
            decisions_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM v2_success_evaluations")
            evaluations_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT AVG(overall_score) FROM v2_success_evaluations")
            avg_score = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT SUM(cost_usd) FROM v2_success_evaluations")
            total_cost = cursor.fetchone()[0]
        
        print(f"\n📊 Generated Demo Data Summary:")
        print(f"  🤖 AI Decisions: {decisions_count}")
        print(f"  📋 Task Evaluations: {evaluations_count}")
        print(f"  🎯 Average Score: {avg_score:.3f}")
        print(f"  💰 Total Cost: ${total_cost:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo data generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_demo_scenarios():
    """Create additional demo scenarios for different use cases"""
    
    print("\n🎬 Creating additional demo scenarios...")
    
    scenarios = [
        {
            "name": "High-Performance Code Generation",
            "tasks": [
                {"type": "code_generation", "complexity": 2.5, "cost": 0.035, "success_rate": 0.95},
                {"type": "code_review", "complexity": 1.8, "cost": 0.020, "success_rate": 0.88},
                {"type": "debugging", "complexity": 2.2, "cost": 0.040, "success_rate": 0.82}
            ]
        },
        {
            "name": "Cost-Optimized Operations", 
            "tasks": [
                {"type": "documentation", "complexity": 0.8, "cost": 0.005, "success_rate": 0.92},
                {"type": "code_generation", "complexity": 1.0, "cost": 0.008, "success_rate": 0.85},
                {"type": "business_analysis", "complexity": 1.2, "cost": 0.012, "success_rate": 0.90}
            ]
        }
    ]
    
    try:
        from shared.kaizen.intelligent_selector import TaskType
        from shared.kaizen.success_evaluator import SuccessEvaluator, TaskResult, TaskOutputType
        
        evaluator = SuccessEvaluator("agent_zero.db")
        
        for scenario in scenarios:
            print(f"  📋 Processing scenario: {scenario['name']}")
            
            for task_config in scenario["tasks"]:
                # Create synthetic task result based on scenario
                task_result = TaskResult(
                    task_id=f"scenario_{scenario['name'].lower().replace(' ', '_')}_{task_config['type']}",
                    task_type=task_config["type"],
                    output_type=TaskOutputType.CODE,
                    output_content=f"# Generated output for {task_config['type']}",
                    expected_requirements=[f"Scenario: {scenario['name']}"],
                    context={"complexity": task_config["complexity"], "scenario": scenario["name"]},
                    execution_time_ms=int(task_config["complexity"] * 2000),
                    cost_usd=task_config["cost"],
                    model_used="gpt-4o" if task_config["cost"] > 0.02 else "llama3.2:3b"
                )
                
                # Force the evaluation score to match scenario
                evaluation = evaluator.evaluate_task(task_result)
                
                # Update the score in database to match scenario expectation
                with sqlite3.connect("agent_zero.db") as conn:
                    conn.execute('''
                        UPDATE v2_success_evaluations
                        SET overall_score = ?, success_level = ?
                        WHERE task_id = ?
                    ''', (
                        task_config["success_rate"],
                        "SUCCESS" if task_config["success_rate"] > 0.85 else "PARTIAL",
                        task_result.task_id
                    ))
        
        print("  ✅ Demo scenarios created successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Demo scenarios creation failed: {e}")
        return False

def main():
    print("🎭 Agent Zero V1 - V2.0 Demo Data Generator")
    print("=" * 50)
    
    if generate_demo_data():
        create_demo_scenarios()
        
        print("\n🎉 Demo data generation complete!")
        print("\n🚀 Try these commands now:")
        print("  python3 -m cli status")
        print("  python3 -m cli kaizen-report --days 7")
        print("  python3 -m cli cost-analysis --threshold 0.01")
        print("  python3 -m cli pattern-discovery")
        print("  python3 -m cli model-reasoning")
        print("  python3 -m cli success-breakdown")
        
        return True
    else:
        print("\n❌ Demo data generation failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)