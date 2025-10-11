#!/usr/bin/env python3
"""
Agent Zero V1 - Context-Aware Agent Selection CLI
Week 43 - Point 2 Implementation - CLI Interface
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# Add paths for proper imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
shared_dir = project_root / "shared"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(shared_dir))
sys.path.insert(0, str(shared_dir / "orchestration"))

# Simple console output
class SimpleConsole:
    def print(self, text, style=None):
        if style == "green":
            print(f"\033[92m{text}\033[0m")
        elif style == "red":
            print(f"\033[91m{text}\033[0m")
        elif style == "yellow":
            print(f"\033[93m{text}\033[0m")
        elif style == "cyan":
            print(f"\033[96m{text}\033[0m")
        elif style == "blue":
            print(f"\033[94m{text}\033[0m")
        elif style == "magenta":
            print(f"\033[95m{text}\033[0m")
        else:
            print(text)
    
    def status(self, text):
        return SimpleStatus(text)

class SimpleStatus:
    def __init__(self, text):
        self.text = text
    
    def __enter__(self):
        print(f"🔄 {self.text}")
        return self
    
    def __exit__(self, *args):
        pass

console = SimpleConsole()

# Import Context-Aware Agent Selection components
AGENT_SELECTION_AVAILABLE = False
try:
    # Import from the context-aware agent selection file
    exec(open(current_dir / "context-aware-agent-selection.py").read())
    
    AGENT_SELECTION_AVAILABLE = True
    console.print("✅ Context-Aware Agent Selection imported successfully", "green")
    
except Exception as e:
    console.print(f"⚠️ Agent selection components not available: {e}", "yellow")
    
    # Fallback classes for basic functionality
    class SelectionStrategy:
        BALANCED = "balanced"
        EXPERTISE_FIRST = "expertise_first"
        AVAILABILITY_FIRST = "availability_first"
        PERFORMANCE_BASED = "performance_based"
    
    class SelectionContext:
        def __init__(self, strategy="balanced", max_team_size=10):
            self.strategy = strategy
            self.max_team_size = max_team_size
    
    class ContextAwareAgentSelector:
        def __init__(self):
            console.print("⚠️ Using fallback agent selector", "yellow")
        
        async def intelligent_team_selection(self, description, domain_context, selection_context):
            return {"message": "Fallback mode - no real selection performed"}

class AgentSelectionCLI:
    """CLI interface for Context-Aware Agent Selection"""
    
    def __init__(self):
        self.selector = None
        if AGENT_SELECTION_AVAILABLE:
            try:
                self.selector = ContextAwareAgentSelector()
                self.selector.populate_demo_agents()
                console.print("✅ Agent Selection system initialized with demo agents", "green")
            except Exception as e:
                console.print(f"⚠️ Failed to initialize agent selector: {e}", "yellow")

    async def select_team(
        self, 
        project_description: str,
        tech_stack: List[str] = None,
        project_type: str = "general",
        strategy: str = "balanced",
        max_team_size: int = 8,
        output_format: str = "table"
    ):
        """Select optimal team for project"""
        
        if not self.selector:
            console.print("❌ Agent selection system not available", "red")
            return
        
        tech_stack = tech_stack or []
        
        console.print(f"\n🧠 Selecting optimal team for project:", "cyan")
        console.print(f"   Description: {project_description}")
        console.print(f"   Tech Stack: {', '.join(tech_stack) if tech_stack else 'General'}")
        console.print(f"   Project Type: {project_type}")
        console.print(f"   Strategy: {strategy}")
        console.print(f"   Max Team Size: {max_team_size}")
        
        # Create contexts
        try:
            domain_context = DomainContext(
                tech_stack=tech_stack,
                project_type=project_type,
                current_phase="development"
            )
            
            # Map string strategy to enum
            strategy_mapping = {
                "balanced": SelectionStrategy.BALANCED,
                "expertise": SelectionStrategy.EXPERTISE_FIRST,
                "availability": SelectionStrategy.AVAILABILITY_FIRST,
                "performance": SelectionStrategy.PERFORMANCE_BASED,
                "collaborative": SelectionStrategy.COLLABORATIVE
            }
            
            selection_context = SelectionContext(
                strategy=strategy_mapping.get(strategy, SelectionStrategy.BALANCED),
                max_team_size=max_team_size
            )
        except Exception as e:
            console.print(f"❌ Failed to create contexts: {e}", "red")
            return
        
        # Perform team selection
        with console.status("Analyzing project and selecting optimal team..."):
            try:
                team_composition = await self.selector.intelligent_team_selection(
                    project_description, domain_context, selection_context
                )
                
                self._display_team_selection_results(team_composition, output_format)
                return team_composition
                
            except Exception as e:
                console.print(f"❌ Team selection failed: {e}", "red")
                import traceback
                traceback.print_exc()

    def _display_team_selection_results(self, team_composition, output_format: str):
        """Display team selection results"""
        
        if output_format == "json":
            # JSON output
            output_data = {
                "team_id": team_composition.team_id,
                "team_score": team_composition.team_score,
                "selected_agents": [
                    {
                        "agent_id": agent.agent_id,
                        "agent_type": agent.agent_type,
                        "expertise": agent.primary_expertise,
                        "current_workload": agent.current_workload,
                        "availability": agent.get_availability()
                    }
                    for agent in team_composition.selected_agents
                ],
                "task_assignments": [
                    {
                        "task_id": assignment.task.id,
                        "task_title": assignment.task.title,
                        "assigned_agent": assignment.assigned_agent.agent_id,
                        "assignment_score": assignment.assignment_score,
                        "confidence": assignment.confidence
                    }
                    for assignment in team_composition.task_assignments
                ],
                "coverage_analysis": team_composition.coverage_analysis,
                "potential_risks": team_composition.potential_risks,
                "recommendations": team_composition.recommendations,
                "timeline": team_composition.estimated_timeline
            }
            print(json.dumps(output_data, indent=2))
            return
        
        # Table/detailed output
        console.print(f"\n🎯 TEAM SELECTION RESULTS", "cyan")
        print("=" * 70)
        
        # Team overview
        console.print(f"\n📊 Team Overview:")
        print(f"   Team ID: {team_composition.team_id}")
        print(f"   Team Score: {team_composition.team_score:.2f}/1.0")
        print(f"   Selected Agents: {len(team_composition.selected_agents)}")
        print(f"   Total Assignments: {len(team_composition.task_assignments)}")
        
        # Selected agents
        console.print(f"\n👥 Selected Team Members:")
        print("-" * 70)
        
        for i, agent in enumerate(team_composition.selected_agents, 1):
            utilization = agent.current_workload / agent.max_workload
            availability = agent.get_availability()
            
            print(f"\n{i}. {agent.agent_id}")
            print(f"   Type: {agent.agent_type}")
            print(f"   Expertise: {', '.join(agent.primary_expertise)}")
            print(f"   Workload: {agent.current_workload:.1f}h / {agent.max_workload:.1f}h ({utilization:.1%})")
            print(f"   Availability: {availability:.1%}")
            
            # Show top capabilities
            top_caps = sorted(agent.capabilities, key=lambda c: c.proficiency_level, reverse=True)[:3]
            if top_caps:
                caps_str = ", ".join([f"{c.name} ({c.proficiency_level:.1%})" for c in top_caps])
                print(f"   Top Skills: {caps_str}")
        
        # Task assignments
        console.print(f"\n📋 Task Assignments:")
        print("-" * 70)
        
        assignment_by_agent = {}
        for assignment in team_composition.task_assignments:
            agent_id = assignment.assigned_agent.agent_id
            if agent_id not in assignment_by_agent:
                assignment_by_agent[agent_id] = []
            assignment_by_agent[agent_id].append(assignment)
        
        for agent_id, assignments in assignment_by_agent.items():
            print(f"\n🤖 {agent_id}:")
            total_hours = sum(a.task.estimated_hours for a in assignments)
            print(f"   Total Load: {total_hours:.1f}h across {len(assignments)} tasks")
            
            for assignment in assignments:
                task = assignment.task
                print(f"   • {task.title} ({task.task_type.value})")
                print(f"     Hours: {task.estimated_hours}h | Priority: {task.priority.value}")
                print(f"     Score: {assignment.assignment_score:.2f} | Confidence: {assignment.confidence:.1%}")
                
                if assignment.reasoning:
                    print(f"     Reasoning: {'; '.join(assignment.reasoning[:2])}")  # Show top 2 reasons
        
        # Coverage analysis
        console.print(f"\n📊 Coverage Analysis:")
        print("-" * 70)
        for agent_type, coverage in team_composition.coverage_analysis.items():
            status = "✅" if coverage >= 0.8 else "⚠️" if coverage >= 0.6 else "❌"
            print(f"   {status} {agent_type}: {coverage:.1%} coverage")
        
        # Risks and recommendations
        if team_composition.potential_risks:
            console.print(f"\n⚠️ Potential Risks:", "yellow")
            for i, risk in enumerate(team_composition.potential_risks, 1):
                print(f"   {i}. {risk}")
        
        if team_composition.recommendations:
            console.print(f"\n💡 Recommendations:", "blue")
            for i, rec in enumerate(team_composition.recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Timeline
        timeline = team_composition.estimated_timeline
        if timeline:
            console.print(f"\n📅 Project Timeline:")
            print(f"   Duration: {timeline.get('total_duration_days', 0)} days")
            print(f"   Start: {timeline.get('start_date', 'TBD')}")
            print(f"   End: {timeline.get('end_date', 'TBD')}")
            
            if timeline.get('milestones'):
                print(f"   Milestones: {len(timeline['milestones'])}")
                for milestone in timeline['milestones'][:3]:  # Show first 3
                    print(f"     • {milestone['phase']} (Day {milestone['day']})")

    async def compare_strategies(
        self, 
        project_description: str,
        tech_stack: List[str] = None,
        project_type: str = "general"
    ):
        """Compare different selection strategies"""
        
        console.print(f"\n🔄 Comparing selection strategies...", "cyan")
        
        strategies = ["balanced", "expertise", "availability", "performance"]
        results = {}
        
        for strategy in strategies:
            console.print(f"\n🧪 Testing strategy: {strategy}", "yellow")
            
            result = await self.select_team(
                project_description, tech_stack, project_type, 
                strategy=strategy, output_format="silent"
            )
            
            if result:
                results[strategy] = {
                    "team_score": result.team_score,
                    "team_size": len(result.selected_agents),
                    "total_coverage": sum(result.coverage_analysis.values()) / len(result.coverage_analysis),
                    "risk_count": len(result.potential_risks)
                }
        
        # Display comparison
        console.print(f"\n📊 Strategy Comparison Results:", "cyan")
        print("=" * 80)
        print(f"{'Strategy':<15} {'Score':<8} {'Size':<6} {'Coverage':<10} {'Risks':<6}")
        print("-" * 80)
        
        for strategy, data in results.items():
            print(f"{strategy:<15} {data['team_score']:<8.2f} {data['team_size']:<6} "
                  f"{data['total_coverage']:<10.1%} {data['risk_count']:<6}")
        
        # Find best strategy
        best_strategy = max(results.keys(), key=lambda s: results[s]['team_score'])
        console.print(f"\n🏆 Best Strategy: {best_strategy} (score: {results[best_strategy]['team_score']:.2f})", "green")

    async def analyze_agent_pool(self):
        """Analyze available agent pool"""
        
        if not self.selector or not hasattr(self.selector, 'agent_profiles'):
            console.print("❌ Agent pool not available", "red")
            return
        
        agents = list(self.selector.agent_profiles.values())
        
        console.print(f"\n📊 Agent Pool Analysis", "cyan")
        print("=" * 60)
        
        # Basic statistics
        total_agents = len(agents)
        avg_workload = sum(a.current_workload for a in agents) / total_agents
        avg_availability = sum(a.get_availability() for a in agents) / total_agents
        
        print(f"Total Agents: {total_agents}")
        print(f"Average Workload: {avg_workload:.1f}h")
        print(f"Average Availability: {avg_availability:.1%}")
        
        # By agent type
        console.print(f"\n👥 Agents by Type:")
        type_counts = {}
        for agent in agents:
            type_counts[agent.agent_type] = type_counts.get(agent.agent_type, 0) + 1
        
        for agent_type, count in sorted(type_counts.items()):
            print(f"   {agent_type}: {count} agents")
        
        # Availability analysis
        console.print(f"\n📈 Availability Analysis:")
        available = len([a for a in agents if a.get_availability() > 0.5])
        overloaded = len([a for a in agents if a.get_availability() < 0.2])
        
        print(f"   Available (>50%): {available} agents")
        print(f"   Busy (20-50%): {total_agents - available - overloaded} agents")
        print(f"   Overloaded (<20%): {overloaded} agents")
        
        # Top performers
        console.print(f"\n🌟 Top Performers:")
        top_performers = sorted(
            agents, 
            key=lambda a: a.performance_history.get('last_month', 0.5), 
            reverse=True
        )[:3]
        
        for i, agent in enumerate(top_performers, 1):
            performance = agent.performance_history.get('last_month', 0.5)
            print(f"   {i}. {agent.agent_id} ({agent.agent_type}) - {performance:.1%}")

def parse_args():
    """Parse command line arguments"""
    if len(sys.argv) < 2:
        print_help()
        return None
    
    command = sys.argv[1]
    
    if command == "select":
        if len(sys.argv) < 3:
            print("❌ Usage: python agent_selection_cli.py select \"project description\" [options]")
            return None
        
        description = sys.argv[2]
        tech_stack = []
        project_type = "general"
        strategy = "balanced"
        max_team_size = 8
        output_format = "table"
        
        # Parse additional arguments
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--tech-stack" and i + 1 < len(sys.argv):
                tech_stack = sys.argv[i + 1].split(",")
                tech_stack = [t.strip() for t in tech_stack]
                i += 2
            elif sys.argv[i] == "--project-type" and i + 1 < len(sys.argv):
                project_type = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--strategy" and i + 1 < len(sys.argv):
                strategy = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--max-team-size" and i + 1 < len(sys.argv):
                max_team_size = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
                output_format = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        
        return {
            "command": "select",
            "description": description,
            "tech_stack": tech_stack,
            "project_type": project_type,
            "strategy": strategy,
            "max_team_size": max_team_size,
            "output": output_format
        }
    
    elif command == "compare":
        if len(sys.argv) < 3:
            print("❌ Usage: python agent_selection_cli.py compare \"project description\" [options]")
            return None
        
        return {
            "command": "compare",
            "description": sys.argv[2],
            "tech_stack": [],
            "project_type": "general"
        }
    
    elif command == "analyze":
        return {"command": "analyze"}
    
    elif command == "demo":
        return {"command": "demo"}
    
    elif command == "help":
        print_help()
        return None
    
    else:
        print(f"❌ Unknown command: {command}")
        print_help()
        return None

def print_help():
    """Print help information"""
    print("""
🧠 Agent Zero V1 - Context-Aware Agent Selection CLI

Commands:
  select "description" [options]     Select optimal team for project
  compare "description" [options]    Compare different selection strategies  
  analyze                           Analyze agent pool
  demo                              Run demonstration
  help                              Show this help

Options for select:
  --tech-stack tech1,tech2          Technology stack (comma-separated)
  --project-type type               Project type
  --strategy strategy               Selection strategy
  --max-team-size N                 Maximum team size
  --output format                   Output format (table, json)

Selection Strategies:
  balanced          Balance expertise and availability (default)
  expertise         Prioritize expertise over availability
  availability      Prioritize availability over expertise  
  performance       Based on historical performance
  collaborative     Optimize for team collaboration

Examples:
  python agent_selection_cli.py select "Create user auth system"
  
  python agent_selection_cli.py select "Build e-commerce API" \\
    --tech-stack FastAPI,PostgreSQL,Redis --project-type api_service \\
    --strategy expertise --max-team-size 6
    
  python agent_selection_cli.py compare "Create social media platform"
  
  python agent_selection_cli.py analyze

Project Types:
  fullstack_web_app, api_service, mobile_app, data_pipeline, ml_project
""")

async def run_demo():
    """Run demonstration of agent selection system"""
    console.print("🚀 Context-Aware Agent Selection Demo", "cyan")
    print("=" * 60)
    
    cli = AgentSelectionCLI()
    
    if not cli.selector:
        console.print("❌ Demo requires agent selection system", "red")
        return
    
    # Demo scenarios
    scenarios = [
        {
            "description": "Create comprehensive e-commerce platform with user management, product catalog, shopping cart, and payment processing",
            "tech_stack": ["FastAPI", "React", "PostgreSQL", "Docker", "Redis"],
            "project_type": "fullstack_web_app",
            "strategy": "balanced"
        },
        {
            "description": "Build high-performance REST API for real-time chat application with WebSocket support",
            "tech_stack": ["FastAPI", "PostgreSQL", "Redis"],
            "project_type": "api_service", 
            "strategy": "performance"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        console.print(f"\n🎯 Demo Scenario {i}/{len(scenarios)}", "cyan")
        console.print(f"Strategy: {scenario['strategy']}", "yellow")
        print("-" * 50)
        
        await cli.select_team(
            scenario["description"],
            scenario["tech_stack"],
            scenario["project_type"],
            scenario["strategy"],
            output_format="table"
        )
        
        if i < len(scenarios):
            input("\n⏵ Press Enter to continue to next scenario...")
    
    # Agent pool analysis
    console.print(f"\n📊 Agent Pool Analysis", "cyan")
    print("-" * 30)
    await cli.analyze_agent_pool()
    
    console.print(f"\n✅ Demo completed!", "green")

async def main():
    """Main CLI function"""
    console.print("🧠 Agent Zero V1 - Context-Aware Agent Selection", "cyan")
    print("=" * 65)
    
    args = parse_args()
    if not args:
        return
    
    cli = AgentSelectionCLI()
    
    if args["command"] == "select":
        await cli.select_team(
            args["description"],
            args["tech_stack"],
            args["project_type"],
            args["strategy"],
            args["max_team_size"],
            args["output"]
        )
    
    elif args["command"] == "compare":
        await cli.compare_strategies(
            args["description"],
            args["tech_stack"],
            args["project_type"]
        )
    
    elif args["command"] == "analyze":
        await cli.analyze_agent_pool()
    
    elif args["command"] == "demo":
        await run_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", "yellow")
    except Exception as e:
        console.print(f"\n❌ Error: {e}", "red")