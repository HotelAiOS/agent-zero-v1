#!/usr/bin/env python3
"""
Agent Zero V1 Command Line Interface
"""
import click
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Agent Zero V1 Command Line Interface"""
    pass

@cli.command()
def start():
    """Start Agent Zero V1 server"""
    click.echo("🚀 Starting Agent Zero V1...")
    import subprocess
    subprocess.run([sys.executable, "run.py"])

@cli.command() 
def status():
    """Check Agent Zero V1 status"""
    click.echo("📊 Agent Zero V1 Status: Ready")

@cli.command()
def agents():
    """List available agents"""
    click.echo("🤖 Available Agents:")
    click.echo("  - default: Basic agent")

if __name__ == "__main__":
    cli()
