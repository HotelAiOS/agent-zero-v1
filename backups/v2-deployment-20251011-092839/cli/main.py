#!/usr/bin/env python3
"""
Agent Zero V1 - Enhanced CLI Commands 
V2.0 Intelligence Layer - Advanced Command Interface

Author: Developer A (Backend Architect)  
Date: 10 października 2025
Linear Issue: A0-28
"""

import argparse
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

import logging
from enhanced_task_commands import task_cli as enhanced_task_cli


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentZeroCLI:
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = Path(db_path)
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
    
    def print_message(self, message: str, style: str = None):
        if self.console and RICH_AVAILABLE:
            self.console.print(message, style=style)
        else:
            print(message)

    def main(self):
        parser = argparse.ArgumentParser(
            prog='a0',
            description='🚀 Agent Zero V1 - Enhanced CLI with V2.0 Intelligence Layer'
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # V2.0 Enhanced Commands
        subparsers.add_parser('status', help='Show system status')
        
        kaizen_parser = subparsers.add_parser('kaizen-report', help='📊 Generate Kaizen report')
        kaizen_parser.add_argument('--days', type=int, default=7, help='Days to analyze')
        kaizen_parser.add_argument('--format', choices=['table', 'json'], default='table')
        
        cost_parser = subparsers.add_parser('cost-analysis', help='💰 Analyze costs')
        cost_parser.add_argument('--threshold', type=float, default=0.02, help='Cost threshold')
        
        subparsers.add_parser('pattern-discovery', help='🔍 Discover patterns')
        subparsers.add_parser('model-reasoning', help='🤖 AI decision explanation')
        subparsers.add_parser('success-breakdown', help='📈 Success analysis')
        
        # V1.0 Legacy Commands
        subparsers.add_parser('run', help='Execute agent task')
        subparsers.add_parser('test', help='Run system tests')
        subparsers.add_parser('deploy', help='Deploy services')
        
        args = parser.parse_args()
        
        if args.command is None:
            parser.print_help()
            return
        
        # Route to handlers
        if args.command == 'status':
            self.handle_status()
        elif args.command == 'kaizen-report':
            self.handle_kaizen_report(args)
        elif args.command == 'cost-analysis':
            self.handle_cost_analysis(args)
        elif args.command == 'pattern-discovery':
            self.handle_pattern_discovery()
        elif args.command == 'model-reasoning':
            self.handle_model_reasoning()
        elif args.command == 'success-breakdown':
            self.handle_success_breakdown()
        elif args.command in ['run', 'test', 'deploy']:
            self.handle_legacy_command(args.command)
        else:
            self.print_message(f"❌ Unknown command: {args.command}", "red")

    def handle_status(self):
        self.print_message("🔍 Agent Zero V1 System Status", "bold cyan")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check V2.0 components
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                v2_tables = [t for t in tables if t.startswith('v2_')]
                
                self.print_message(f"Database: ✅ Connected ({self.db_path})", "green")
                self.print_message(f"V2.0 Tables: {len(v2_tables)} found", "green")
                
                for table in v2_tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    self.print_message(f"  {table}: {count} records", "cyan")
                    
        except Exception as e:
            self.print_message(f"❌ Status check failed: {e}", "red")

    def handle_kaizen_report(self, args):
        self.print_message("📊 Generating Kaizen Report...", "cyan")
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_tasks,
                        AVG(overall_score) as avg_score,
                        SUM(cost_usd) as total_cost,
                        AVG(execution_time_ms) as avg_latency
                    FROM v2_success_evaluations 
                    WHERE timestamp BETWEEN ? AND ?
                ''', (start_date, end_date))
                
                row = cursor.fetchone()
                
                if row and row[0] > 0:
                    total_tasks, avg_score, total_cost, avg_latency = row
                    
                    if args.format == 'json':
                        report = {
                            'period_days': args.days,
                            'total_tasks': total_tasks,
                            'avg_score': round(avg_score or 0, 3),
                            'total_cost': round(total_cost or 0, 4),
                            'avg_latency_ms': round(avg_latency or 0, 0)
                        }
                        print(json.dumps(report, indent=2))
                    else:
                        self.print_message(f"\n📈 Kaizen Report ({args.days} days)", "bold")
                        self.print_message(f"Total Tasks: {total_tasks}")
                        self.print_message(f"Average Score: {avg_score or 0:.3f}")
                        self.print_message(f"Total Cost: ${total_cost or 0:.4f}")
                        self.print_message(f"Average Latency: {avg_latency or 0:.0f}ms")
                else:
                    self.print_message("📊 No data available for the specified period", "yellow")
                    
        except Exception as e:
            self.print_message(f"❌ Failed to generate report: {e}", "red")

    def handle_cost_analysis(self, args):
        self.print_message("💰 Analyzing Cost Optimization...", "cyan")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT 
                        model_used,
                        AVG(cost_usd) as avg_cost,
                        COUNT(*) as frequency,
                        AVG(overall_score) as avg_score
                    FROM v2_success_evaluations
                    WHERE cost_usd > ?
                    GROUP BY model_used
                    ORDER BY AVG(cost_usd) DESC
                ''', (args.threshold,))
                
                results = cursor.fetchall()
                
                if results:
                    self.print_message(f"\n💸 High-Cost Tasks (threshold: ${args.threshold:.3f})", "bold")
                    
                    for row in results:
                        model, avg_cost, freq, avg_score = row
                        potential_savings = avg_cost * freq * 0.3
                        
                        self.print_message(f"Model: {model}")
                        self.print_message(f"  Avg Cost: ${avg_cost:.4f}")
                        self.print_message(f"  Frequency: {freq}")
                        self.print_message(f"  Avg Score: {avg_score:.3f}")
                        self.print_message(f"  Potential Savings: ${potential_savings:.4f}")
                        self.print_message("")
                        
                    self.print_message("💡 Recommendations:", "yellow")
                    self.print_message("• Consider using local Ollama models for routine tasks")
                    self.print_message("• Batch similar requests to reduce overhead")
                    self.print_message("• Use lighter models for simple tasks")
                else:
                    self.print_message("💰 No high-cost tasks found", "green")
                    
        except Exception as e:
            self.print_message(f"❌ Cost analysis failed: {e}", "red")

    def handle_pattern_discovery(self):
        self.print_message("🔍 Discovering Success Patterns...", "cyan")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT 
                        model_used,
                        task_type,
                        COUNT(*) as frequency,
                        AVG(overall_score) as avg_score
                    FROM v2_success_evaluations
                    WHERE overall_score >= 0.8
                    GROUP BY model_used, task_type
                    HAVING COUNT(*) >= 3
                    ORDER BY AVG(overall_score) DESC, COUNT(*) DESC
                ''')
                
                results = cursor.fetchall()
                
                if results:
                    self.print_message("\n🎯 Successful Model-Task Patterns", "bold")
                    
                    for row in results:
                        model, task_type, freq, avg_score = row
                        self.print_message(f"{model} → {task_type}")
                        self.print_message(f"  Frequency: {freq}, Success Rate: {avg_score:.3f}")
                else:
                    self.print_message("🔍 No significant patterns found", "yellow")
                    
        except Exception as e:
            self.print_message(f"❌ Pattern discovery failed: {e}", "red")

    def handle_model_reasoning(self):
        self.print_message("🤖 Recent AI Decisions...", "cyan")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT 
                        timestamp, task_type, recommended_model,
                        confidence_score, reasoning
                    FROM v2_model_decisions
                    ORDER BY timestamp DESC
                    LIMIT 5
                ''')
                
                results = cursor.fetchall()
                
                if results:
                    self.print_message("\n🤖 Recent AI Decisions", "bold")
                    
                    for row in results:
                        timestamp, task_type, model, confidence, reasoning = row
                        self.print_message(f"Time: {timestamp}")
                        self.print_message(f"Task: {task_type} → {model}")
                        self.print_message(f"Confidence: {confidence:.3f}")
                        self.print_message(f"Reasoning: {reasoning}")
                        self.print_message("")
                else:
                    self.print_message("🤖 No recent decisions found", "yellow")
                    
        except Exception as e:
            self.print_message(f"❌ Reasoning analysis failed: {e}", "red")

    def handle_success_breakdown(self):
        self.print_message("📈 Success Dimension Analysis...", "cyan")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT 
                        AVG(correctness_score) as avg_correctness,
                        AVG(efficiency_score) as avg_efficiency,
                        AVG(cost_score) as avg_cost,
                        AVG(latency_score) as avg_latency
                    FROM v2_success_evaluations
                ''')
                
                row = cursor.fetchone()
                
                if row and any(x is not None for x in row):
                    correctness, efficiency, cost, latency = row
                    
                    self.print_message("\n📊 Average Dimension Scores", "bold")
                    self.print_message(f"Correctness: {correctness or 0:.3f} (50% weight)")
                    self.print_message(f"Efficiency: {efficiency or 0:.3f} (20% weight)")  
                    self.print_message(f"Cost: {cost or 0:.3f} (15% weight)")
                    self.print_message(f"Latency: {latency or 0:.3f} (15% weight)")
                    
                    # Overall weighted score
                    if all(x is not None for x in row):
                        overall = (correctness * 0.5 + efficiency * 0.2 + 
                                 cost * 0.15 + latency * 0.15)
                        self.print_message(f"\nWeighted Overall: {overall:.3f}", "green")
                else:
                    self.print_message("📊 No success data available", "yellow")
                    
        except Exception as e:
            self.print_message(f"❌ Success analysis failed: {e}", "red")

    def handle_legacy_command(self, command):
        self.print_message(f"🚀 Agent Zero V1 - {command.title()}", "bold cyan")
        self.print_message(f"Note: V1.0 {command} command (V2.0 enhancements available)", "yellow")

def main():
    cli = AgentZeroCLI()
    cli.main()

if __name__ == "__main__":
    main()
