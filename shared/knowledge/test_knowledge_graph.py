"""
Test Knowledge Graph - Sprawdź zapisywanie i wyszukiwanie wiedzy
"""
import asyncio
import logging
from knowledge_graph import knowledge_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_knowledge_graph():
    """Test pełnego workflow Knowledge Graph"""
    
    print("\n" + "="*70)
    print("🧪 TESTING KNOWLEDGE GRAPH")
    print("="*70 + "\n")
    
    # 1. Połącz
    print("📝 Krok 1: Łączenie z Neo4j...")
    await knowledge_graph.connect()
    print("   ✅ Połączono\n")
    
    # 2. Zapisz pierwsze zadanie (symulacja z wczorajszego testu)
    print("📝 Krok 2: Zapisywanie pierwszej generacji kodu...")
    task1_id = await knowledge_graph.record_code_generation(
        agent_id="backend_001",
        task_description="Create a FastAPI endpoint for user registration",
        generated_code="""from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
import bcrypt

app = FastAPI()

class UserCreate(BaseModel):
    email: EmailStr
    password: str

@app.post("/register")
async def register_user(user: UserCreate):
    # Hash password
    hashed = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt())
    # Save to database...
    return {"message": "User registered"}
""",
        model_used="deepseek-coder:33b",
        processing_time=1211.95,
        success=True
    )
    print(f"   ✅ Zapisano: {task1_id}\n")
    
    # 3. Zapisz drugie zadanie
    print("📝 Krok 3: Zapisywanie drugiej generacji kodu...")
    task2_id = await knowledge_graph.record_code_generation(
        agent_id="backend_001",
        task_description="Create user authentication API with JWT tokens",
        generated_code="""from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
import jwt

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/login")
async def login(username: str, password: str):
    # Verify credentials...
    token = jwt.encode({"sub": username}, "secret")
    return {"access_token": token}
    
@app.get("/protected")
async def protected_route(token: str = Depends(oauth2_scheme)):
    # Verify token...
    return {"message": "Access granted"}
""",
        model_used="deepseek-coder:33b",
        processing_time=987.34,
        success=True
    )
    print(f"   ✅ Zapisano: {task2_id}\n")
    
    # 4. Znajdź podobne zadania
    print("📝 Krok 4: Wyszukiwanie podobnych zadań...")
    similar = await knowledge_graph.find_similar_tasks(
        task_description="user registration endpoint",
        limit=5
    )
    
    print(f"   🔍 Znaleziono {len(similar)} podobnych zadań:\n")
    for i, task in enumerate(similar, 1):
        print(f"   {i}. {task['description']}")
        print(f"      Agent: {task['agent_id']}")
        print(f"      Model: {task['model_used']}")
        print(f"      Czas: {task['processing_time']:.2f}s")
        print(f"      Język: {task['language']}")
        print(f"      Kod (preview): {task['code'][:80]}...")
        print()
    
    # 5. Statystyki agenta
    print("📝 Krok 5: Statystyki agenta backend_001...")
    stats = await knowledge_graph.get_agent_stats("backend_001")
    
    print(f"   📊 Statystyki:")
    print(f"      Total zadań: {stats['total_tasks']}")
    print(f"      Sukces: {stats['successful_tasks']}")
    print(f"      Success rate: {stats['success_rate']:.1f}%")
    print(f"      Średni czas: {stats['avg_processing_time']:.2f}s")
    print(f"      Wygenerowane linie: {stats['total_code_lines']}")
    print()
    
    # 6. Zamknij
    await knowledge_graph.close()
    
    print("="*70)
    print("✅ KNOWLEDGE GRAPH TEST COMPLETED!")
    print("="*70 + "\n")
    
    print("📊 Podsumowanie:")
    print("   ✅ Połączenie z Neo4j")
    print("   ✅ Zapisywanie kodu")
    print("   ✅ Wyszukiwanie podobnych zadań")
    print("   ✅ Statystyki agentów")
    print("\n🎉 System ma teraz PAMIĘĆ i może się UCZYĆ!\n")


if __name__ == "__main__":
    asyncio.run(test_knowledge_graph())
