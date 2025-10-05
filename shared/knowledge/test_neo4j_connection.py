"""
Test Neo4j Connection - Sprawdź czy baza działa
"""
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_neo4j_connection():
    """Test połączenia z Neo4j"""
    
    # Credentials z docker-compose.yml
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "agent-pass"
    
    try:
        logger.info("🔌 Łączę z Neo4j...")
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Test query
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            value = result.single()["test"]
            
            if value == 1:
                logger.info("✅ Neo4j connection SUCCESS!")
                logger.info(f"   URI: {uri}")
                logger.info(f"   Username: {username}")
                return True
            
    except Exception as e:
        logger.error(f"❌ Neo4j connection FAILED: {e}")
        logger.error("   Sprawdź czy Neo4j container działa: docker ps | grep neo4j")
        return False
    
    finally:
        if 'driver' in locals():
            driver.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING NEO4J CONNECTION")
    print("="*70 + "\n")
    
    success = test_neo4j_connection()
    
    print("\n" + "="*70)
    if success:
        print("✅ NEO4J READY FOR KNOWLEDGE GRAPH!")
    else:
        print("❌ NEO4J NOT READY - Check Docker")
    print("="*70 + "\n")
