#!/usr/bin/env python3
"""
SQLite to Neo4j Migration Script
Agent Zero V2.0 - Data Migration for 40% Performance Improvement
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.knowledge.graph_integration_v2 import SQLiteToNeo4jMigrator, AgentZeroGraphSchema
from shared.knowledge.neo4j_client import Neo4jClient

async def main():
    """Run complete migration from SQLite to Neo4j"""
    print("🔄 Agent Zero V2.0 - Data Migration to Neo4j")
    print("=" * 50)
    
    try:
        # Initialize Neo4j client
        neo4j_client = Neo4jClient()
        
        # Initialize schema
        print("1️⃣ Initializing Neo4j V2.0 schema...")
        schema = AgentZeroGraphSchema(neo4j_client)
        schema_result = await schema.initialize_v2_schema()
        
        if schema_result['status'] == 'success':
            print(f"   ✅ Schema initialized in {schema_result['setup_time_ms']:.1f}ms")
        else:
            print(f"   ❌ Schema initialization failed: {schema_result['error']}")
            return 1
        
        # Run migration
        print("\n2️⃣ Migrating data from SQLite...")
        migrator = SQLiteToNeo4jMigrator(neo4j_client=neo4j_client)
        migration_result = await migrator.migrate_all_data()
        
        if migration_result['status'] == 'success':
            print(f"   ✅ Migration completed in {migration_result['migration_time_seconds']:.1f}s")
            print(f"   📊 Experiences migrated: {migration_result['experiences'].get('migrated', 0)}")
            print(f"   🏗️ Models updated: {migration_result['models'].get('models_updated', 0)}")
            print(f"   🎯 Patterns created: {migration_result['patterns'].get('patterns_created', 0)}")
            return 0
        else:
            print(f"   ❌ Migration failed: {migration_result['error']}")
            return 1
            
    except Exception as e:
        print(f"❌ Migration error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
