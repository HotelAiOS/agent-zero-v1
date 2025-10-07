#!/usr/bin/env python3
"""
Agent Zero V1 - Neo4j Health Check Script
Comprehensive monitoring and verification script for Neo4j service health

This script provides detailed health monitoring for the Neo4j service
in the Agent Zero V1 multi-agent system, including connection testing,
performance metrics, and service validation.

Author: Agent Zero V1 Development Team
Version: 1.0.0
Date: 2025-10-07
Usage: python healthcheck-neo4j.py [--detailed] [--json] [--timeout=30]
"""

import sys
import time
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import requests
import subprocess
import socket
from pathlib import Path

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Warning: neo4j package not available. Some checks will be skipped.")

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class Neo4jHealthChecker:
    """Comprehensive Neo4j health checking utility."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize health checker with configuration."""
        self.config = config or self._load_default_config()
        self.results = {}
        self.logger = self._setup_logging()
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from environment or defaults."""
        import os
        return {
            "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
            "neo4j_password": os.getenv("NEO4J_PASSWORD", "agent_zero_2024!"),
            "neo4j_database": os.getenv("NEO4J_DATABASE", "neo4j"),
            "http_url": "http://localhost:7474",
            "container_name": "agent-zero-neo4j",
            "timeout": 30,
            "detailed": False,
            "json_output": False
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for health checker."""
        logger = logging.getLogger("neo4j_healthcheck")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def print_status(self, message: str, status: str = "INFO", color: str = Colors.WHITE):
        """Print colored status message."""
        if not self.config.get("json_output", False):
            timestamp = datetime.now().strftime("%H:%M:%S")
            status_colors = {
                "PASS": Colors.GREEN,
                "FAIL": Colors.RED,
                "WARN": Colors.YELLOW,
                "INFO": Colors.BLUE
            }
            status_color = status_colors.get(status, color)
            print(f"[{timestamp}] {status_color}[{status}]{Colors.END} {message}")
    
    def check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity to Neo4j ports."""
        self.print_status("Checking network connectivity...", "INFO")
        
        results = {}
        
        # Check HTTP port (7474)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', 7474))
            sock.close()
            
            http_reachable = result == 0
            results["http_port_reachable"] = http_reachable
            
            if http_reachable:
                self.print_status("✓ HTTP port 7474 is reachable", "PASS")
            else:
                self.print_status("✗ HTTP port 7474 is not reachable", "FAIL")
                
        except Exception as e:
            results["http_port_reachable"] = False
            results["http_port_error"] = str(e)
            self.print_status(f"✗ HTTP port check failed: {e}", "FAIL")
        
        # Check Bolt port (7687)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', 7687))
            sock.close()
            
            bolt_reachable = result == 0
            results["bolt_port_reachable"] = bolt_reachable
            
            if bolt_reachable:
                self.print_status("✓ Bolt port 7687 is reachable", "PASS")
            else:
                self.print_status("✗ Bolt port 7687 is not reachable", "FAIL")
                
        except Exception as e:
            results["bolt_port_reachable"] = False
            results["bolt_port_error"] = str(e)
            self.print_status(f"✗ Bolt port check failed: {e}", "FAIL")
        
        return results
    
    def check_http_interface(self) -> Dict[str, Any]:
        """Check Neo4j HTTP interface accessibility."""
        self.print_status("Checking HTTP interface...", "INFO")
        
        results = {}
        
        try:
            start_time = time.time()
            response = requests.get(
                self.config["http_url"],
                timeout=self.config["timeout"]
            )
            response_time = (time.time() - start_time) * 1000
            
            results["http_accessible"] = response.status_code == 200
            results["http_status_code"] = response.status_code
            results["http_response_time_ms"] = round(response_time, 2)
            
            if response.status_code == 200:
                self.print_status(
                    f"✓ HTTP interface accessible (Response: {response_time:.1f}ms)", 
                    "PASS"
                )
                
                # Try to get server info
                try:
                    # Check for Neo4j-specific headers or content
                    if "neo4j" in response.text.lower():
                        results["neo4j_detected"] = True
                        self.print_status("✓ Neo4j server detected in response", "PASS")
                    else:
                        results["neo4j_detected"] = False
                        self.print_status("⚠ Neo4j not clearly identified in response", "WARN")
                except:
                    results["neo4j_detected"] = None
            else:
                self.print_status(
                    f"✗ HTTP interface returned status {response.status_code}", 
                    "FAIL"
                )
                
        except requests.exceptions.ConnectionError:
            results["http_accessible"] = False
            results["error"] = "Connection refused"
            self.print_status("✗ HTTP interface connection refused", "FAIL")
            
        except requests.exceptions.Timeout:
            results["http_accessible"] = False
            results["error"] = "Request timeout"
            self.print_status("✗ HTTP interface request timeout", "FAIL")
            
        except Exception as e:
            results["http_accessible"] = False
            results["error"] = str(e)
            self.print_status(f"✗ HTTP interface check failed: {e}", "FAIL")
        
        return results
    
    def check_bolt_connection(self) -> Dict[str, Any]:
        """Check Neo4j Bolt connection."""
        self.print_status("Checking Bolt connection...", "INFO")
        
        results = {}
        
        if not NEO4J_AVAILABLE:
            results["bolt_accessible"] = None
            results["error"] = "Neo4j Python driver not available"
            self.print_status("⚠ Neo4j Python driver not available", "WARN")
            return results
        
        try:
            start_time = time.time()
            
            driver = GraphDatabase.driver(
                self.config["neo4j_uri"],
                auth=(self.config["neo4j_user"], self.config["neo4j_password"]),
                connection_timeout=self.config["timeout"],
                max_retry_time=10
            )
            
            # Verify connectivity
            driver.verify_connectivity()
            connection_time = (time.time() - start_time) * 1000
            
            results["bolt_accessible"] = True
            results["bolt_connection_time_ms"] = round(connection_time, 2)
            
            self.print_status(
                f"✓ Bolt connection successful (Connect: {connection_time:.1f}ms)", 
                "PASS"
            )
            
            # Test basic query
            try:
                with driver.session(database=self.config["neo4j_database"]) as session:
                    query_start = time.time()
                    result = session.run("RETURN 1 as test")
                    test_value = result.single()["test"]
                    query_time = (time.time() - query_start) * 1000
                    
                    if test_value == 1:
                        results["basic_query_success"] = True
                        results["query_response_time_ms"] = round(query_time, 2)
                        self.print_status(
                            f"✓ Basic query executed (Query: {query_time:.1f}ms)", 
                            "PASS"
                        )
                    else:
                        results["basic_query_success"] = False
                        self.print_status("✗ Basic query returned unexpected result", "FAIL")
                        
            except Exception as e:
                results["basic_query_success"] = False
                results["query_error"] = str(e)
                self.print_status(f"✗ Basic query failed: {e}", "FAIL")
            
            driver.close()
            
        except Exception as e:
            results["bolt_accessible"] = False
            results["error"] = str(e)
            self.print_status(f"✗ Bolt connection failed: {e}", "FAIL")
        
        return results
    
    def check_docker_container(self) -> Dict[str, Any]:
        """Check Docker container status."""
        self.print_status("Checking Docker container...", "INFO")
        
        results = {}
        
        if not DOCKER_AVAILABLE:
            results["docker_available"] = False
            self.print_status("⚠ Docker library not available", "WARN")
            return results
        
        try:
            client = docker.from_env()
            container = client.containers.get(self.config["container_name"])
            
            results["container_found"] = True
            results["container_status"] = container.status
            results["container_id"] = container.short_id
            
            if container.status == "running":
                self.print_status(f"✓ Container '{self.config['container_name']}' is running", "PASS")
                
                # Get container stats if detailed mode
                if self.config.get("detailed", False):
                    try:
                        stats = container.stats(stream=False)
                        
                        # CPU usage calculation
                        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                                   stats['precpu_stats']['cpu_usage']['total_usage']
                        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                      stats['precpu_stats']['system_cpu_usage']
                        
                        cpu_percent = 0.0
                        if system_delta > 0:
                            cpu_percent = (cpu_delta / system_delta) * 100.0
                        
                        # Memory usage
                        mem_usage = stats['memory_stats']['usage']
                        mem_limit = stats['memory_stats']['limit']
                        mem_percent = (mem_usage / mem_limit) * 100.0
                        
                        results["cpu_percent"] = round(cpu_percent, 2)
                        results["memory_usage_mb"] = round(mem_usage / 1024 / 1024, 2)
                        results["memory_percent"] = round(mem_percent, 2)
                        
                        self.print_status(
                            f"  CPU: {cpu_percent:.1f}%, Memory: {mem_percent:.1f}% "
                            f"({mem_usage/1024/1024:.1f}MB)", 
                            "INFO"
                        )
                        
                    except Exception as e:
                        self.print_status(f"⚠ Could not get container stats: {e}", "WARN")
                
            else:
                self.print_status(
                    f"✗ Container '{self.config['container_name']}' status: {container.status}", 
                    "FAIL"
                )
                
        except docker.errors.NotFound:
            results["container_found"] = False
            self.print_status(f"✗ Container '{self.config['container_name']}' not found", "FAIL")
            
        except Exception as e:
            results["docker_error"] = str(e)
            self.print_status(f"✗ Docker check failed: {e}", "FAIL")
        
        return results
    
    def check_database_health(self) -> Dict[str, Any]:
        """Check database health and performance."""
        self.print_status("Checking database health...", "INFO")
        
        results = {}
        
        if not NEO4J_AVAILABLE:
            results["database_health"] = None
            self.print_status("⚠ Skipping database health (driver not available)", "WARN")
            return results
        
        try:
            driver = GraphDatabase.driver(
                self.config["neo4j_uri"],
                auth=(self.config["neo4j_user"], self.config["neo4j_password"])
            )
            
            with driver.session(database=self.config["neo4j_database"]) as session:
                # Node count
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = result.single()["node_count"]
                results["node_count"] = node_count
                
                # Relationship count
                result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = result.single()["rel_count"]
                results["relationship_count"] = rel_count
                
                # Label information
                result = session.run("CALL db.labels()")
                labels = [record["label"] for record in result]
                results["labels"] = labels
                results["label_count"] = len(labels)
                
                # Relationship types
                result = session.run("CALL db.relationshipTypes()")
                rel_types = [record["relationshipType"] for record in result]
                results["relationship_types"] = rel_types
                results["relationship_type_count"] = len(rel_types)
                
                self.print_status(
                    f"✓ Database health: {node_count} nodes, {rel_count} relationships, "
                    f"{len(labels)} labels", 
                    "PASS"
                )
                
                if self.config.get("detailed", False):
                    self.print_status(f"  Labels: {', '.join(labels[:5])}" + 
                                    ("..." if len(labels) > 5 else ""), "INFO")
                    self.print_status(f"  Relationship types: {', '.join(rel_types[:3])}" + 
                                    ("..." if len(rel_types) > 3 else ""), "INFO")
            
            driver.close()
            
        except Exception as e:
            results["database_health"] = False
            results["error"] = str(e)
            self.print_status(f"✗ Database health check failed: {e}", "FAIL")
        
        return results
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive results."""
        self.print_status("Starting comprehensive Neo4j health check...", "INFO", Colors.CYAN)
        start_time = time.time()
        
        # Run all checks
        checks = {
            "network": self.check_network_connectivity(),
            "http_interface": self.check_http_interface(),
            "bolt_connection": self.check_bolt_connection(),
            "docker_container": self.check_docker_container(),
            "database_health": self.check_database_health()
        }
        
        # Calculate overall health
        total_time = time.time() - start_time
        
        # Determine overall status
        critical_failures = []
        warnings = []
        
        if not checks["network"].get("http_port_reachable", False):
            critical_failures.append("HTTP port not reachable")
        if not checks["network"].get("bolt_port_reachable", False):
            critical_failures.append("Bolt port not reachable")
        if not checks["http_interface"].get("http_accessible", False):
            critical_failures.append("HTTP interface not accessible")
        if not checks["bolt_connection"].get("bolt_accessible", False):
            critical_failures.append("Bolt connection failed")
        
        if checks["docker_container"].get("container_status") != "running":
            critical_failures.append("Docker container not running")
        
        overall_healthy = len(critical_failures) == 0
        
        # Compile final results
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_healthy": overall_healthy,
            "total_check_time_seconds": round(total_time, 2),
            "critical_failures": critical_failures,
            "warnings": warnings,
            "checks": checks,
            "summary": {
                "network_ok": checks["network"].get("http_port_reachable", False) and 
                            checks["network"].get("bolt_port_reachable", False),
                "http_ok": checks["http_interface"].get("http_accessible", False),
                "bolt_ok": checks["bolt_connection"].get("bolt_accessible", False),
                "container_ok": checks["docker_container"].get("container_status") == "running",
                "database_ok": checks["database_health"].get("node_count") is not None
            }
        }
        
        # Print summary
        if not self.config.get("json_output", False):
            self.print_summary(final_results)
        
        return final_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print human-readable summary of health check results."""
        print(f"\n{Colors.CYAN}{'='*60}{Colors.END}")
        print(f"{Colors.CYAN}Neo4j Health Check Summary{Colors.END}")
        print(f"{Colors.CYAN}{'='*60}{Colors.END}")
        
        # Overall status
        if results["overall_healthy"]:
            print(f"\n{Colors.GREEN}✓ Overall Status: HEALTHY{Colors.END}")
        else:
            print(f"\n{Colors.RED}✗ Overall Status: UNHEALTHY{Colors.END}")
        
        print(f"Check completed in {results['total_check_time_seconds']}s")
        
        # Summary table
        summary = results["summary"]
        print(f"\n{Colors.BOLD}Component Status:{Colors.END}")
        print(f"  Network Connectivity: {self._status_symbol(summary['network_ok'])}")
        print(f"  HTTP Interface:       {self._status_symbol(summary['http_ok'])}")
        print(f"  Bolt Connection:      {self._status_symbol(summary['bolt_ok'])}")
        print(f"  Docker Container:     {self._status_symbol(summary['container_ok'])}")
        print(f"  Database Health:      {self._status_symbol(summary['database_ok'])}")
        
        # Critical failures
        if results["critical_failures"]:
            print(f"\n{Colors.RED}{Colors.BOLD}Critical Issues:{Colors.END}")
            for failure in results["critical_failures"]:
                print(f"  {Colors.RED}• {failure}{Colors.END}")
        
        # Database info
        if "database_health" in results["checks"] and results["checks"]["database_health"]:
            db_info = results["checks"]["database_health"]
            if "node_count" in db_info:
                print(f"\n{Colors.BOLD}Database Information:{Colors.END}")
                print(f"  Nodes: {db_info['node_count']}")
                print(f"  Relationships: {db_info['relationship_count']}")
                print(f"  Labels: {db_info['label_count']}")
                print(f"  Relationship Types: {db_info['relationship_type_count']}")
        
        print(f"\n{Colors.CYAN}{'='*60}{Colors.END}")
    
    def _status_symbol(self, status: bool) -> str:
        """Return colored status symbol."""
        if status:
            return f"{Colors.GREEN}✓ OK{Colors.END}"
        else:
            return f"{Colors.RED}✗ FAIL{Colors.END}"


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Agent Zero V1 - Neo4j Health Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python healthcheck-neo4j.py                    # Basic health check
  python healthcheck-neo4j.py --detailed         # Detailed health check
  python healthcheck-neo4j.py --json             # JSON output
  python healthcheck-neo4j.py --timeout=60       # Custom timeout
        """
    )
    
    parser.add_argument(
        "--detailed", 
        action="store_true",
        help="Show detailed information including container stats"
    )
    
    parser.add_argument(
        "--json",
        action="store_true", 
        help="Output results in JSON format"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Connection timeout in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--uri",
        type=str,
        help="Neo4j URI (default: bolt://localhost:7687)"
    )
    
    parser.add_argument(
        "--user",
        type=str,
        help="Neo4j username (default: neo4j)"
    )
    
    parser.add_argument(
        "--password", 
        type=str,
        help="Neo4j password (default: agent_zero_2024!)"
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = {}
    
    if args.uri:
        config["neo4j_uri"] = args.uri
    if args.user:
        config["neo4j_user"] = args.user
    if args.password:
        config["neo4j_password"] = args.password
    
    config["timeout"] = args.timeout
    config["detailed"] = args.detailed
    config["json_output"] = args.json
    
    # Run health check
    checker = Neo4jHealthChecker(config)
    results = checker.run_comprehensive_check()
    
    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    
    # Exit with appropriate status code
    sys.exit(0 if results["overall_healthy"] else 1)


if __name__ == "__main__":
    main()