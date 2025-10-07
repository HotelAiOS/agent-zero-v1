#!/usr/bin/env python3
"""
Neo4j Health Monitor - Agent Zero V1
Real-time health monitoring and diagnostics
Arch Linux + Fish Shell Compatible

Author: Agent Zero Development Team
Date: 2025-10-07
Version: 1.0.0
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import docker
from neo4j import GraphDatabase


class Neo4jHealthMonitor:
    """Real-time Neo4j health monitoring system."""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "agent_zero_neo4j_dev"
        self.log_file = Path("/tmp/neo4j_health.log")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def check_container_health(self) -> Dict:
        """Check Docker container health status."""
        try:
            container = self.docker_client.containers.get("agent-zero-neo4j")
            
            # Get container stats
            stats = container.stats(stream=False)
            
            health_status = {
                "status": container.status,
                "running": container.status == "running",
                "created": container.attrs['Created'],
                "started": container.attrs['State'].get('StartedAt', 'N/A'),
                "cpu_usage": self._calculate_cpu_percentage(stats),
                "memory_usage": self._calculate_memory_usage(stats),
                "network_io": self._get_network_io(stats),
                "ports": {
                    "7474": self._check_port(7474),
                    "7687": self._check_port(7687)
                },
                "health_check": container.attrs['State'].get('Health', {})
            }
            
            return health_status
            
        except docker.errors.NotFound:
            return {
                "status": "not_found",
                "running": False,
                "error": "Container 'agent-zero-neo4j' not found"
            }
        except Exception as e:
            return {
                "status": "error",
                "running": False,
                "error": str(e)
            }

    def _calculate_cpu_percentage(self, stats: Dict) -> float:
        """Calculate CPU usage percentage."""
        try:
            cpu_stats = stats['cpu_stats']
            precpu_stats = stats['precpu_stats']
            
            cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
            system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']
            number_cpus = cpu_stats['online_cpus']
            
            if system_delta > 0 and cpu_delta > 0:
                return (cpu_delta / system_delta) * number_cpus * 100.0
            return 0.0
        except (KeyError, ZeroDivisionError):
            return 0.0

    def _calculate_memory_usage(self, stats: Dict) -> Dict:
        """Calculate memory usage statistics."""
        try:
            memory = stats['memory_stats']
            usage = memory['usage']
            limit = memory['limit']
            
            return {
                "usage_mb": round(usage / (1024 * 1024), 2),
                "limit_mb": round(limit / (1024 * 1024), 2),
                "percentage": round((usage / limit) * 100, 2)
            }
        except KeyError:
            return {"usage_mb": 0, "limit_mb": 0, "percentage": 0}

    def _get_network_io(self, stats: Dict) -> Dict:
        """Get network I/O statistics."""
        try:
            networks = stats['networks']
            total_rx = sum(net['rx_bytes'] for net in networks.values())
            total_tx = sum(net['tx_bytes'] for net in networks.values())
            
            return {
                "rx_mb": round(total_rx / (1024 * 1024), 2),
                "tx_mb": round(total_tx / (1024 * 1024), 2)
            }
        except KeyError:
            return {"rx_mb": 0, "tx_mb": 0}

    def _check_port(self, port: int) -> bool:
        """Check if a port is accessible."""
        try:
            result = subprocess.run([
                "nc", "-z", "localhost", str(port)
            ], capture_output=True, timeout=2)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def check_neo4j_connectivity(self) -> Dict:
        """Check Neo4j database connectivity and performance."""
        try:
            driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
                connection_acquisition_timeout=5
            )
            
            # Verify connectivity
            start_time = time.time()
            driver.verify_connectivity()
            connection_time = time.time() - start_time
            
            # Run diagnostic queries
            with driver.session() as session:
                # Basic connectivity test
                start_time = time.time()
                result = session.run("RETURN 1 as test")
                record = result.single()
                query_time = time.time() - start_time
                
                # Get database info
                db_info = session.run("""
                CALL dbms.components() 
                YIELD name, versions, edition 
                RETURN name, versions[0] as version, edition
                """).data()
                
                # Get node/relationship counts
                counts = session.run("""
                MATCH (n) 
                OPTIONAL MATCH ()-[r]-() 
                RETURN count(DISTINCT n) as nodes, count(DISTINCT r) as relationships
                """).single()
                
                # Get memory usage
                memory_info = session.run("CALL dbms.memory.tracking()").data()
                
            driver.close()
            
            return {
                "connected": True,
                "connection_time_ms": round(connection_time * 1000, 2),
                "query_time_ms": round(query_time * 1000, 2),
                "database_info": db_info,
                "node_count": counts["nodes"] if counts else 0,
                "relationship_count": counts["relationships"] if counts else 0,
                "memory_tracking": memory_info,
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "last_check": datetime.now().isoformat()
            }

    def get_neo4j_logs(self, lines: int = 50) -> List[str]:
        """Get recent Neo4j container logs."""
        try:
            container = self.docker_client.containers.get("agent-zero-neo4j")
            logs = container.logs(tail=lines, timestamps=True)
            return logs.decode('utf-8').split('\n')
        except Exception as e:
            return [f"Error getting logs: {e}"]

    def run_comprehensive_health_check(self) -> Dict:
        """Run comprehensive health check and return results."""
        self.logger.info("Running comprehensive Neo4j health check...")
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "container_health": self.check_container_health(),
            "connectivity": self.check_neo4j_connectivity(),
            "recent_logs": self.get_neo4j_logs(20)
        }
        
        # Determine overall health status
        container_ok = health_report["container_health"].get("running", False)
        connectivity_ok = health_report["connectivity"].get("connected", False)
        
        health_report["overall_status"] = "healthy" if (container_ok and connectivity_ok) else "unhealthy"
        health_report["issues"] = self._identify_issues(health_report)
        
        return health_report

    def _identify_issues(self, health_report: Dict) -> List[str]:
        """Identify potential issues from health report."""
        issues = []
        
        container_health = health_report["container_health"]
        connectivity = health_report["connectivity"]
        
        # Container issues
        if not container_health.get("running"):
            issues.append("Container is not running")
            
        if container_health.get("memory_usage", {}).get("percentage", 0) > 80:
            issues.append("High memory usage detected")
            
        # Connectivity issues
        if not connectivity.get("connected"):
            error_type = connectivity.get("error_type", "Unknown")
            issues.append(f"Database connectivity failed: {error_type}")
            
        if connectivity.get("connection_time_ms", 0) > 1000:
            issues.append("Slow database connection detected")
            
        # Port issues
        ports = container_health.get("ports", {})
        if not ports.get("7474"):
            issues.append("HTTP port 7474 is not accessible")
        if not ports.get("7687"):
            issues.append("Bolt port 7687 is not accessible")
            
        return issues

    def print_health_summary(self, health_report: Dict) -> None:
        """Print formatted health summary."""
        status = health_report["overall_status"]
        status_color = "\033[92m" if status == "healthy" else "\033[91m"
        reset_color = "\033[0m"
        
        print("\n" + "="*60)
        print("🔍 NEO4J HEALTH MONITOR - COMPREHENSIVE REPORT")
        print("="*60)
        print(f"Overall Status: {status_color}{status.upper()}{reset_color}")
        print(f"Check Time: {health_report['timestamp']}")
        
        # Container info
        container = health_report["container_health"]
        print(f"\n📦 CONTAINER STATUS:")
        print(f"  Running: {'✅' if container.get('running') else '❌'}")
        if "memory_usage" in container:
            mem = container["memory_usage"]
            print(f"  Memory: {mem['usage_mb']}MB / {mem['limit_mb']}MB ({mem['percentage']}%)")
        
        # Connectivity info
        conn = health_report["connectivity"]
        print(f"\n🔗 DATABASE CONNECTIVITY:")
        print(f"  Connected: {'✅' if conn.get('connected') else '❌'}")
        if conn.get("connected"):
            print(f"  Connection Time: {conn.get('connection_time_ms', 0)}ms")
            print(f"  Nodes: {conn.get('node_count', 0)}")
            print(f"  Relationships: {conn.get('relationship_count', 0)}")
        
        # Issues
        issues = health_report.get("issues", [])
        if issues:
            print(f"\n⚠️  IDENTIFIED ISSUES:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print(f"\n✅ NO ISSUES DETECTED")
            
        print("="*60)

    async def monitor_continuously(self, interval: int = 30) -> None:
        """Continuously monitor Neo4j health."""
        self.logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        
        try:
            while True:
                health_report = self.run_comprehensive_health_check()
                
                # Log status
                status = health_report["overall_status"]
                self.logger.info(f"Health check: {status}")
                
                if health_report["issues"]:
                    for issue in health_report["issues"]:
                        self.logger.warning(f"Issue detected: {issue}")
                
                # Save detailed report
                report_file = Path("/tmp/neo4j_health_report.json")
                with open(report_file, "w") as f:
                    json.dump(health_report, f, indent=2, default=str)
                
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")


def main():
    """Main function."""
    monitor = Neo4jHealthMonitor()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "check":
            # Single health check
            report = monitor.run_comprehensive_health_check()
            monitor.print_health_summary(report)
            
            # Exit with error code if unhealthy
            sys.exit(0 if report["overall_status"] == "healthy" else 1)
            
        elif command == "monitor":
            # Continuous monitoring
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            asyncio.run(monitor.monitor_continuously(interval))
            
        elif command == "json":
            # JSON output
            report = monitor.run_comprehensive_health_check()
            print(json.dumps(report, indent=2, default=str))
            
        else:
            print("Usage: neo4j_health_check.py [check|monitor [interval]|json]")
            sys.exit(1)
    else:
        # Default: single check with formatted output
        report = monitor.run_comprehensive_health_check()
        monitor.print_health_summary(report)


if __name__ == "__main__":
    main()