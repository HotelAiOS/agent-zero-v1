"""
Test integracji Agent Factory z LLM
"""

import sys
from pathlib import Path

# Dodaj parent directory do path
sys.path.append(str(Path(__file__).parent.parent))

from agent_factory.factory import AgentFactory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_agent_with_llm():
    """Test wykonania zadania przez agenta z użyciem LLM"""
    
    print("\n" + "="*70)
    print("🧪 TEST: Agent Factory + LLM Integration")
    print("="*70 + "\n")
    
    # Krok 1: Utwórz fabrykę
    print("1️⃣  Tworzenie Agent Factory...")
    factory = AgentFactory()
    print(f"   ✅ Fabryka gotowa ({len(factory.list_available_types())} typów agentów)")
    
    # Krok 2: Utwórz agenta Backend
    print("\n2️⃣  Tworzenie agenta Backend...")
    agent = factory.create_agent('backend')
    
    if not agent:
        print("   ❌ Nie udało się utworzyć agenta")
        return False
    
    print(f"   ✅ Agent utworzony: {agent.agent_id}")
    print(f"      Stan: {agent.state.value}")
    print(f"      Typ: {agent.agent_type}")
    
    # Krok 3: Przypisz zadanie
    print("\n3️⃣  Wykonywanie zadania...")
    task = {
        'description': 'Napisz funkcję Python do walidacji adresu email. '
                      'Funkcja powinna zwracać True jeśli email jest poprawny, '
                      'False w przeciwnym razie. Użyj regex.',
        'context': {
            'language': 'python',
            'style': 'professional'
        }
    }
    
    print(f"   Zadanie: {task['description'][:60]}...")
    print(f"   Wywołanie LLM (może zająć 10-30s)...")
    
    # Wykonaj zadanie
    result = agent.execute_task(task)
    
    # Krok 4: Sprawdź wynik
    print("\n4️⃣  Analiza wyniku...")
    
    if result['success']:
        print("   ✅ Zadanie wykonane pomyślnie!")
        print(f"      Tokeny: {result.get('tokens_used', 0)}")
        print(f"      Czas: {result.get('response_time', 0):.2f}s")
        print(f"\n   📄 Wygenerowany kod:")
        print("   " + "-"*66)
        
        output = result['output']
        if output:
            # Pokaż max 20 linii
            lines = output.split('\n')[:20]
            for line in lines:
                print(f"   {line}")
            if len(output.split('\n')) > 20:
                print(f"   ... ({len(output.split('\n')) - 20} more lines)")
        else:
            print("   (brak kodu w odpowiedzi)")
        
        print("   " + "-"*66)
        
    else:
        print("   ❌ Zadanie nie powiodło się")
        print(f"      Błąd: {result.get('error', 'unknown')}")
        return False
    
    # Krok 5: Sprawdź metryki agenta
    print("\n5️⃣  Metryki agenta:")
    metrics = agent.metrics
    print(f"   Ukończone: {metrics.tasks_completed}")
    print(f"   Błędy: {metrics.tasks_failed}")
    print(f"   Tokeny użyte: {metrics.total_tokens_used}")
    print(f"   Średni czas: {metrics.average_response_time:.2f}s")
    
    print("\n" + "="*70)
    print("✅ TEST ZAKOŃCZONY SUKCESEM!")
    print("="*70 + "\n")
    
    return True


if __name__ == '__main__':
    try:
        success = test_agent_with_llm()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test error: {e}", exc_info=True)
        exit(1)
