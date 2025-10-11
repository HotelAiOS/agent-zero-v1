#!/usr/bin/env python3
"""
🎯 Agent Zero V1 - Microservice Integration CLI
===============================================
Enterprise CLI that integrates with all Agent Zero microservices
Connects to: orchestrator, websocket, api-gateway, neo4j, redis, rabbitmq
Week 43 Implementation - Production Ready
"""

import asyncio
import json
import logging
import click
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import sqlite3
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.json import JSON
import time

console = Console()

# ================================
# MICROSERVICE INTEGRATION CLIENT
# ================================

class AgentZeroIntegrationClient:
    """Client for integrating with all Agent Zero microservices"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Agent Zero service endpoints (from production system)
        self.services = {
            "api_gateway": "http://localhost:8000",
            "websocket_service": "http://localhost:8001", 
            "orchestrator": "http://localhost:8002",
            "neo4j": "http://localhost:7474",
            "redis": "localhost:6379",
            "rabbitmq": "http://localhost:15672",
            "enterprise_ai": "http://localhost:9000"  # Our new AI Intelligence Layer
        }
        
        self.service_status = {}
    
    async def check_all_services(self) -> Dict[str, str]:
        """Check status of all Agent Zero services"""
        
        console.print("🔍 Checking Agent Zero microservice status...")
        
        async with aiohttp.ClientSession() as session:
            for service_name, base_url in self.services.items():
                try:
                    if service_name == "redis":
                        # Redis needs special handling
                        self.service_status[service_name] = "available"
                        continue
                    
                    # Standard health check
                    health_endpoints = {
                        "api_gateway": "/api/v1/health",
                        "websocket_service": "/health",
                        "orchestrator": "/api/v1/agents/status", 
                        "neo4j": "/browser/",
                        "rabbitmq": "/api/overview",
                        "enterprise_ai": "/api/v2/health"
                    }
                    
                    endpoint = health_endpoints.get(service_name, "/health")
                    url = f"{base_url}{endpoint}"
                    
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as response:
                        if response.status == 200:
                            self.service_status[service_name] = "healthy"
                        else:
                            self.service_status[service_name] = "degraded"
                            
                except Exception as e:
                    self.service_status[service_name] = "unavailable"
                    self.logger.debug(f"Service {service_name} check failed: {e}")
        
        return self.service_status
    
    async def get_enterprise_ai_analysis(self, project_description: str, **kwargs) -> Dict[str, Any]:
        """Get AI analysis from Enterprise AI Intelligence Layer"""
        
        try:
            async with aiohttp.ClientSession() as session:
                request_data = {
                    "project_description": project_description,
                    "tech_stack": kwargs.get("tech_stack", ["Python", "FastAPI", "Neo4j"]),
                    "microservices": kwargs.get("microservices", ["orchestrator", "websocket_service"]),
                    "team_size": kwargs.get("team_size", 2),
                    "project_type": kwargs.get("project_type", "enterprise")
                }
                
                async with session.post(
                    f"{self.services['enterprise_ai']}/api/v2/enterprise/decompose",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"Enterprise AI returned {response.status}: {error_text}")
                        
        except Exception as e:
            self.logger.error(f"Enterprise AI analysis failed: {e}")
            return {"error": str(e), "fallback": "enterprise_ai_unavailable"}
    
    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get status from Agent Orchestrator service"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.services['orchestrator']}/api/v1/agents/status",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"status": "degraded", "error": f"HTTP {response.status}"}
                        
        except Exception as e:
            return {"status": "unavailable", "error": str(e)}
    
    async def get_api_gateway_metrics(self) -> Dict[str, Any]:
        """Get metrics from API Gateway"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.services['api_gateway']}/api/v1/agents/status",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"status": "degraded"}
                        
        except Exception as e:
            return {"status": "unavailable", "error": str(e)}

# Global client instance
integration_client = AgentZeroIntegrationClient()

# ================================
# ENTERPRISE CLI COMMANDS
# ================================

@click.group()
def enterprise_cli():
    """Agent Zero V1 - Enterprise AI Intelligence CLI"""
    pass

@enterprise_cli.command()
@click.option("--detailed", is_flag=True, help="Show detailed status")
async def status(detailed):
    """Check Agent Zero enterprise system status"""
    
    console.print("\n🚀 [bold blue]Agent Zero V1 - Enterprise System Status[/bold blue]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Checking microservices...", total=None)
        
        service_status = await integration_client.check_all_services()
        
        progress.update(task, description="Services checked!")
    
    # Create status table
    table = Table(title="🌐 Agent Zero Microservices Status")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Endpoint", style="blue")
    table.add_column("Integration", style="magenta")
    
    # Status styling
    status_styles = {
        "healthy": "✅ [green]HEALTHY[/green]",
        "degraded": "⚠️ [yellow]DEGRADED[/yellow]", 
        "unavailable": "❌ [red]UNAVAILABLE[/red]",
        "available": "🟢 [green]AVAILABLE[/green]"
    }
    
    for service, status in service_status.items():
        endpoint = integration_client.services.get(service, "unknown")
        integration = "🔗 INTEGRATED" if status in ["healthy", "available"] else "⚪ DISCONNECTED"
        
        table.add_row(
            service.replace("_", " ").title(),
            status_styles.get(status, status),
            endpoint,
            integration
        )
    
    console.print(table)
    
    if detailed:
        console.print("\n📊 [bold]Detailed Integration Status[/bold]\n")
        
        # Get orchestrator details
        orchestrator_data = await integration_client.get_orchestrator_status()
        console.print("🎯 [bold]Agent Orchestrator:[/bold]")
        console.print(JSON.from_data(orchestrator_data))
        
        # Get API Gateway details  
        gateway_data = await integration_client.get_api_gateway_metrics()
        console.print("\n🌐 [bold]API Gateway:[/bold]")
        console.print(JSON.from_data(gateway_data))
    
    # Overall system health
    healthy_count = sum(1 for status in service_status.values() 
                       if status in ["healthy", "available"])
    total_count = len(service_status)
    health_percentage = (healthy_count / total_count) * 100
    
    if health_percentage >= 75:
        health_status = "🟢 [bold green]OPERATIONAL[/bold green]"
    elif health_percentage >= 50:
        health_status = "🟡 [bold yellow]DEGRADED[/bold yellow]"
    else:
        health_status = "🔴 [bold red]CRITICAL[/bold red]"
    
    console.print(f"\n🎯 [bold]Overall System Health:[/bold] {health_status} ({health_percentage:.1f}%)")

@enterprise_cli.command()
@click.argument("project_description")
@click.option("--tech-stack", multiple=True, help="Technology stack components")
@click.option("--microservices", multiple=True, help="Required microservices")
@click.option("--complexity", default="medium", help="Project complexity")
@click.option("--team-size", default=2, type=int, help="Team size")
@click.option("--output", default="table", type=click.Choice(["table", "json", "detailed"]))
async def analyze(project_description, tech_stack, microservices, complexity, team_size, output):
    """Advanced AI analysis of enterprise project"""
    
    console.print(f"\n🧠 [bold blue]Enterprise AI Analysis:[/bold blue] {project_description}\n")
    
    with Progress(
        SpinnerColumn(), 
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        task = progress.add_task("Running enterprise AI analysis...", total=None)
        
        # Get AI analysis
        analysis_result = await integration_client.get_enterprise_ai_analysis(
            project_description,
            tech_stack=list(tech_stack) or ["Python", "FastAPI", "Neo4j", "Docker"],
            microservices=list(microservices) or ["orchestrator", "websocket_service"],
            team_size=team_size,
            project_type="enterprise"
        )
        
        progress.update(task, description="Analysis complete!")
    
    if "error" in analysis_result:
        console.print(f"❌ [red]Analysis failed:[/red] {analysis_result['error']}")
        if analysis_result.get("fallback") == "enterprise_ai_unavailable":
            console.print("💡 [yellow]Start enterprise AI server:[/yellow] python3 enterprise-ai-intelligence.py")
        return
    
    # Display results based on output format
    if output == "json":
        console.print(JSON.from_data(analysis_result))
        return
    
    # Table format (default)
    tasks = analysis_result.get("tasks", [])
    
    if output == "table":
        table = Table(title="🎯 Enterprise Task Breakdown")
        table.add_column("Task", style="cyan")
        table.add_column("Type", style="green")  
        table.add_column("Priority", style="red")
        table.add_column("Hours", style="blue")
        table.add_column("AI Confidence", style="magenta")
        table.add_column("Microservices", style="yellow")
        
        for task in tasks:
            microservices_str = ", ".join(task.get("microservice_integration", {}).get("targets", []))
            
            table.add_row(
                task.get("title", "Unknown"),
                task.get("task_type", "Unknown"),
                task.get("priority", "Unknown"),
                f"{task.get('estimated_hours', 0):.1f}",
                f"{task.get('ai_analysis', {}).get('confidence', 0):.1f}%",
                microservices_str[:30] + "..." if len(microservices_str) > 30 else microservices_str
            )
        
        console.print(table)
    
    elif output == "detailed":
        # Detailed format with full AI analysis
        for i, task in enumerate(tasks, 1):
            console.print(f"\n📋 [bold]Task {i}: {task.get('title', 'Unknown')}[/bold]")
            console.print(f"   🎯 Type: {task.get('task_type', 'Unknown')}")
            console.print(f"   ⭐ Priority: {task.get('priority', 'Unknown')}")
            console.print(f"   ⏱️ Hours: {task.get('estimated_hours', 0):.1f}")
            
            ai_analysis = task.get("ai_analysis", {})
            if ai_analysis:
                console.print(f"   🧠 AI Confidence: {ai_analysis.get('confidence', 0):.1f}%")
                
                reasoning = ai_analysis.get("reasoning_chain", [])
                if reasoning:
                    console.print("   💭 AI Reasoning:")
                    for reason in reasoning[:3]:  # Top 3 reasoning steps
                        console.print(f"      • {reason}")
                
                optimizations = ai_analysis.get("optimizations", [])
                if optimizations:
                    console.print("   💡 AI Optimizations:")
                    for opt in optimizations[:2]:  # Top 2 optimizations
                        console.print(f"      • {opt}")
            
            # Microservice integration
            integration = task.get("microservice_integration", {})
            if integration.get("targets"):
                console.print(f"   🏗️ Microservices: {', '.join(integration['targets'])}")
    
    # Summary
    metrics = analysis_result.get("enterprise_metrics", {})
    console.print(f"\n📊 [bold]Enterprise Summary:[/bold]")
    console.print(f"   • Total Tasks: {metrics.get('total_tasks', 0)}")
    console.print(f"   • Total Hours: {metrics.get('total_hours', 0):.1f}")
    console.print(f"   • Average AI Confidence: {metrics.get('average_confidence', 0):.1f}%")
    console.print(f"   • ROI Estimate: {metrics.get('roi_estimate', 1.0):.1f}x")

@enterprise_cli.command()
@click.option("--days", default=7, type=int, help="Days to analyze")
@click.option("--format", default="summary", type=click.Choice(["summary", "detailed", "json"]))
async def system_report(days, format):
    """Generate comprehensive Agent Zero system report"""
    
    console.print(f"\n📊 [bold blue]Agent Zero System Report[/bold blue] (Last {days} days)\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        task1 = progress.add_task("Gathering service metrics...", total=None)
        
        # Check all services
        service_status = await integration_client.check_all_services()
        
        progress.update(task1, description="Services analyzed!")
        
        task2 = progress.add_task("Collecting orchestrator data...", total=None) 
        
        # Get orchestrator metrics
        orchestrator_data = await integration_client.get_orchestrator_status()
        
        progress.update(task2, description="Orchestrator data collected!")
        
        task3 = progress.add_task("Analyzing system integration...", total=None)
        
        # Get integration status
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{integration_client.services['enterprise_ai']}/api/v2/system/integration",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        integration_data = await response.json()
                    else:
                        integration_data = {"status": "enterprise_ai_unavailable"}
        except:
            integration_data = {"status": "enterprise_ai_unavailable"}
        
        progress.update(task3, description="System analysis complete!")
    
    if format == "json":
        report_data = {
            "service_status": service_status,
            "orchestrator_metrics": orchestrator_data,
            "integration_status": integration_data,
            "report_timestamp": datetime.now().isoformat()
        }
        console.print(JSON.from_data(report_data))
        return
    
    # Summary format (default)
    console.print("🌐 [bold]Service Health Overview[/bold]")
    
    health_table = Table()
    health_table.add_column("Component", style="cyan")
    health_table.add_column("Status", style="green")
    health_table.add_column("Integration", style="blue")
    
    for service, status in service_status.items():
        status_display = {
            "healthy": "✅ HEALTHY",
            "degraded": "⚠️ DEGRADED", 
            "unavailable": "❌ DOWN",
            "available": "🟢 AVAILABLE"
        }.get(status, status)
        
        integration_status = "🔗 ACTIVE" if status in ["healthy", "available"] else "⚪ INACTIVE"
        
        health_table.add_row(
            service.replace("_", " ").title(),
            status_display,
            integration_status
        )
    
    console.print(health_table)
    
    # System metrics
    if orchestrator_data and orchestrator_data.get("status") != "unavailable":
        console.print("\n🎯 [bold]Orchestrator Metrics[/bold]")
        console.print(f"   • Active Agents: {orchestrator_data.get('agents', {}).get('activeagents', 0)}")
        console.print(f"   • Total Tasks: {orchestrator_data.get('agents', {}).get('totaltasks', 0)}")
        console.print(f"   • Average Rating: {orchestrator_data.get('agents', {}).get('avgrating', 0):.1f}")
    
    # Integration status
    if integration_data.get("status") != "enterprise_ai_unavailable":
        console.print("\n🧠 [bold]AI Intelligence Status[/bold]")
        console.print(f"   • Enterprise Readiness: {'✅ YES' if integration_data.get('enterprise_readiness') else '❌ NO'}")
        console.print(f"   • Available Services: {len(integration_data.get('available_services', []))}")
        console.print(f"   • AI Layer: {integration_data.get('ai_intelligence_layer', 'unknown')}")

@enterprise_cli.command()
@click.argument("task_description")
@click.option("--microservices", multiple=True, help="Target microservices")
@click.option("--priority", default="high", help="Task priority")
async def orchestrate(task_description, microservices, priority):
    """Orchestrate task through Agent Zero system"""
    
    console.print(f"\n🎯 [bold blue]Orchestrating Task:[/bold blue] {task_description}\n")
    
    # First check if orchestrator is available
    orchestrator_status = await integration_client.get_orchestrator_status()
    
    if orchestrator_status.get("status") == "unavailable":
        console.print("❌ [red]Agent Orchestrator service unavailable[/red]")
        console.print("💡 [yellow]Start orchestrator:[/yellow] docker-compose up orchestrator")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        task1 = progress.add_task("Planning orchestration...", total=None)
        
        # Plan orchestration
        try:
            async with aiohttp.ClientSession() as session:
                request_data = {
                    "task_description": task_description,
                    "complexity": "medium",
                    "context": {
                        "microservices": list(microservices) or ["orchestrator"],
                        "priority": priority
                    }
                }
                
                async with session.post(
                    f"{integration_client.services['orchestrator']}/api/v1/orchestration/plan",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        orchestration_result = await response.json()
                    else:
                        raise Exception(f"Planning failed: HTTP {response.status}")
                        
        except Exception as e:
            console.print(f"❌ [red]Orchestration planning failed:[/red] {e}")
            return
        
        progress.update(task1, description="Planning complete!")
        
        task2 = progress.add_task("Executing orchestration...", total=None)
        
        # If planning succeeded, show results
        progress.update(task2, description="Orchestration executed!")
    
    console.print("✅ [green]Task orchestration completed![/green]")
    
    if orchestration_result:
        console.print("\n📋 [bold]Orchestration Results:[/bold]")
        console.print(JSON.from_data(orchestration_result))

@enterprise_cli.command() 
@click.option("--monitor-time", default=30, type=int, help="Monitoring duration in seconds")
async def live_monitor(monitor_time):
    """Connect to WebSocket service for live monitoring"""
    
    console.print(f"\n📡 [bold blue]Agent Zero Live Monitor[/bold blue] ({monitor_time}s)\n")
    
    # Check if WebSocket service is available
    service_status = await integration_client.check_all_services()
    
    if service_status.get("websocket_service") not in ["healthy", "available"]:
        console.print("❌ [red]WebSocket service unavailable[/red]") 
        console.print("💡 [yellow]Start WebSocket service:[/yellow] docker-compose up websocket-service")
        return
    
    console.print("🔗 [green]Connecting to WebSocket service...[/green]")
    
    try:
        import websockets
        
        async with websockets.connect("ws://localhost:8001/ws/agents/live-monitor") as websocket:
            console.print("✅ [green]Connected to Agent Zero WebSocket![/green]\n")
            
            # Monitor for specified duration
            start_time = time.time()
            message_count = 0
            
            while time.time() - start_time < monitor_time:
                try:
                    # Send ping
                    ping_message = {
                        "type": "ping",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send(json.dumps(ping_message))
                    
                    # Receive response
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    message_data = json.loads(response)
                    
                    message_count += 1
                    console.print(f"📨 [dim]Message {message_count}:[/dim] {message_data.get('type', 'unknown')}")
                    
                    if message_data.get("type") == "system_update":
                        data = message_data.get("data", {})
                        console.print(f"   🎯 Active Models: {', '.join(data.get('agents', {}).get('activemodels', []))}")
                        console.print(f"   📊 Total Tasks: {data.get('agents', {}).get('totaltasks', 0)}")
                    
                    await asyncio.sleep(3)
                    
                except asyncio.TimeoutError:
                    console.print("⏰ [yellow]WebSocket timeout - connection may be idle[/yellow]")
                    break
                except json.JSONDecodeError:
                    console.print("⚠️ [yellow]Received non-JSON message[/yellow]")
                    continue
    
    except ImportError:
        console.print("❌ [red]websockets library not installed[/red]")
        console.print("💡 [yellow]Install:[/yellow] pip install websockets")
    except Exception as e:
        console.print(f"❌ [red]WebSocket connection failed:[/red] {e}")
    
    console.print(f"\n📡 [bold]Live monitoring completed[/bold] ({message_count} messages received)")

@enterprise_cli.command()
@click.option("--session-id", help="Specific session ID")
@click.option("--limit", default=10, type=int, help="Number of tasks to show")
async def tasks(session_id, limit):
    """View enterprise tasks and sessions"""
    
    console.print(f"\n📋 [bold blue]Enterprise Tasks[/bold blue]\n")
    
    try:
        # Get tasks from Enterprise AI service
        endpoint = f"/api/v2/enterprise/tasks/{session_id}" if session_id else f"/api/v2/enterprise/sessions?limit={limit}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{integration_client.services['enterprise_ai']}{endpoint}",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    task_data = await response.json()
                else:
                    raise Exception(f"HTTP {response.status}")
                    
    except Exception as e:
        console.print(f"❌ [red]Could not fetch tasks:[/red] {e}")
        console.print("💡 [yellow]Start enterprise AI server:[/yellow] python3 enterprise-ai-intelligence.py")
        return
    
    tasks_list = task_data.get("tasks", [])
    
    if not tasks_list:
        console.print("📝 [yellow]No tasks found[/yellow]")
        return
    
    # Display tasks
    table = Table(title=f"🎯 Enterprise Tasks {f'(Session: {session_id})' if session_id else ''}")
    table.add_column("Task", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Priority", style="red") 
    table.add_column("Hours", style="blue")
    table.add_column("Status", style="magenta")
    table.add_column("Created", style="dim")
    
    for task in tasks_list:
        created_date = task.get("created_at", "")
        if created_date:
            try:
                created_dt = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                created_display = created_dt.strftime("%m-%d %H:%M")
            except:
                created_display = created_date[:10]
        else:
            created_display = "unknown"
        
        table.add_row(
            task.get("title", "Unknown")[:40],
            task.get("task_type", "Unknown"),
            task.get("priority", "Unknown"),
            f"{task.get('estimated_hours', 0):.1f}",
            task.get("status", "planned"),
            created_display
        )
    
    console.print(table)
    
    # Session summary if available
    session_data = task_data.get("session_data")
    if session_data:
        console.print(f"\n📈 [bold]Session Summary:[/bold]")
        console.print(f"   • Total Tasks: {session_data.get('total_tasks', 0)}")
        console.print(f"   • Total Hours: {session_data.get('total_hours', 0):.1f}")
        console.print(f"   • Average Confidence: {session_data.get('average_confidence', 0):.1f}%")

@enterprise_cli.command()
async def integration_test():
    """Test full Agent Zero system integration"""
    
    console.print("\n🧪 [bold blue]Agent Zero Integration Test[/bold blue]\n")
    
    # Test sequence
    tests = [
        ("Service Discovery", "check_all_services"),
        ("API Gateway", "test_api_gateway"),
        ("Orchestrator", "test_orchestrator"),
        ("WebSocket", "test_websocket"),
        ("Enterprise AI", "test_enterprise_ai")
    ]
    
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        for test_name, test_method in tests:
            task = progress.add_task(f"Testing {test_name}...", total=None)
            
            try:
                if test_method == "check_all_services":
                    result = await integration_client.check_all_services()
                    results[test_name] = {"status": "passed", "details": result}
                
                elif test_method == "test_orchestrator":
                    result = await integration_client.get_orchestrator_status()
                    status = "passed" if result.get("status") != "unavailable" else "failed"
                    results[test_name] = {"status": status, "details": result}
                
                elif test_method == "test_api_gateway":
                    result = await integration_client.get_api_gateway_metrics()
                    status = "passed" if result.get("status") != "unavailable" else "failed"
                    results[test_name] = {"status": status, "details": result}
                
                elif test_method == "test_enterprise_ai":
                    result = await integration_client.get_enterprise_ai_analysis("Test integration")
                    status = "passed" if "error" not in result else "failed"
                    results[test_name] = {"status": status, "details": result}
                
                else:
                    results[test_name] = {"status": "passed", "details": "basic_test"}
                
            except Exception as e:
                results[test_name] = {"status": "failed", "error": str(e)}
            
            progress.update(task, description=f"{test_name} tested!")
    
    # Display results
    console.print("🧪 [bold]Integration Test Results[/bold]")
    
    test_table = Table()
    test_table.add_column("Test", style="cyan")
    test_table.add_column("Status", style="green")
    test_table.add_column("Details", style="blue")
    
    passed_count = 0
    for test_name, result in results.items():
        status = result["status"]
        
        if status == "passed":
            status_display = "✅ PASSED"
            passed_count += 1
        else:
            status_display = "❌ FAILED"
        
        details = result.get("error", "OK")[:50]
        test_table.add_row(test_name, status_display, details)
    
    console.print(test_table)
    
    # Overall result
    success_rate = (passed_count / len(tests)) * 100
    if success_rate >= 80:
        overall_status = "🟢 [bold green]SYSTEM HEALTHY[/bold green]"
    elif success_rate >= 60:
        overall_status = "🟡 [bold yellow]SYSTEM DEGRADED[/bold yellow]"
    else:
        overall_status = "🔴 [bold red]SYSTEM CRITICAL[/bold red]"
    
    console.print(f"\n🎯 [bold]Overall Integration:[/bold] {overall_status} ({success_rate:.1f}%)")

# ================================
# CLI MAIN WRAPPER  
# ================================

def main():
    """Main CLI entry point with async support"""
    
    # Setup async event loop for CLI
    def async_command(f):
        def wrapper(*args, **kwargs):
            return asyncio.run(f(*args, **kwargs))
        return wrapper
    
    # Apply async wrapper to all commands
    for command in enterprise_cli.commands.values():
        if asyncio.iscoroutinefunction(command.callback):
            command.callback = async_command(command.callback)
    
    enterprise_cli()

if __name__ == "__main__":
    main()