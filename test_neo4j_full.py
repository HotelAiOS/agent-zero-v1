"""
Test Neo4j Knowledge Graph - Full Functionality
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "shared"))

from knowledge import Neo4jClient

def main():
    print("🧪 Testing Neo4j Knowledge Graph\n")
    
    with Neo4jClient() as client:
        # 1. Create agent
        print("1️⃣ Creating agent...")
        client.create_agent_node("backend_1", "backend", ["python", "fastapi", "postgresql"])
        agent = client.get_agent("backend_1")
        print(f"   ✅ Agent: {agent['agent_type']}, capabilities: {agent['capabilities']}\n")
        
        # 2. Create task
        print("2️⃣ Creating task...")
        client.create_task_node("task_001", "Build REST API", "backend_1", project_id="project_alpha", complexity=7)
        print("   ✅ Task created\n")
        
        # 3. Complete task
        print("3️⃣ Completing task...")
        client.complete_task("task_001", success=True, outcome="API built successfully with FastAPI")
        print("   ✅ Task marked as completed\n")
        
        # 4. Store experience
        print("4️⃣ Storing experience...")
        exp_id = client.store_experience(
            "backend_1",
            context="Building REST API with FastAPI and PostgreSQL",
            outcome="Successfully created scalable API with proper error handling",
            success=True,
            metadata={"framework": "FastAPI", "database": "PostgreSQL"}
        )
        print(f"   ✅ Experience stored: {exp_id}\n")
        
        # 5. Get agent stats
        print("5️⃣ Getting agent statistics...")
        stats = client.get_agent_stats("backend_1")
        print(f"   Agent Type: {stats['agent_type']}")
        print(f"   Total Tasks: {stats['total_tasks']}")
        print(f"   Successful Tasks: {stats['successful_tasks']}")
        print(f"   Experiences: {stats['experiences']}\n")
        
        # 6. Search similar experiences
        print("6️⃣ Searching similar experiences...")
        similar = client.get_similar_experiences("backend_1", ["API", "FastAPI"], limit=3)
        print(f"   ✅ Found {len(similar)} similar experiences\n")
        
        print("✅ Neo4j Knowledge Graph test complete!")

if __name__ == "__main__":
    main()
