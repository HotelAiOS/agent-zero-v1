import subprocess
import sqlite3
import time
import asyncio
import websockets
from services.ai_router.src.router.orchestrator import AIOrchestrator
from pathlib import Path

DB_PATH = Path.home() / ".agent-zero/tracker.db"


def run_cli_command(command_args):
    """
    Uruchamia CLI z podanymi argumentami.
    Zwraca stdout (tekst odpowiedzi).
    """
    result = subprocess.run(
        ["python3", "cli/__main__.py"] + command_args,
        capture_output=True, text=True, input="5\n"  # Wysyła 5 jako rating automatycznie
    )
    print(f"\n[CLI CMD {' '.join(command_args)}]:\n{result.stdout}")
    return result.stdout


def test_feedback_storage():
    """Sprawdza czy feedback jest w bazie"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM feedback")
    count = cur.fetchone()[0]
    print(f"Feedback entries in DB: {count}")
    assert count > 0, "Brak wpisów feedback w DB"
    conn.close()


def test_compare_models_output():
    """Sprawdza czy compare-models działa poprawnie"""
    output = run_cli_command(["compare-models"])
    assert "Model Performance" in output, "Brak tabeli wyników Model Performance"


async def test_websocket_echo():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        test_msg = "ping"
        await websocket.send(test_msg)
        resp = await websocket.recv()
        print(f"WebSocket response: {resp}")
        assert "ping" in resp, "Brak odpowiedzi echo z WebSocket"


def test_orchestrator_model_selection():
    orchestrator = AIOrchestrator()
    model_chat = orchestrator.get_best_model_for_task("chat")
    model_code = orchestrator.get_best_model_for_task("code_generation")
    print(f"Selected model for 'chat': {model_chat}")
    print(f"Selected model for 'code_generation': {model_code}")
    assert model_chat in ["llama3.2-3b", "qwen2.5-coder:7b"], "Niepoprawny model dla chat"
    assert model_code in ["qwen2.5-coder:7b", "llama3.2-3b"], "Niepoprawny model dla code_generation"


if __name__ == "__main__":
    # 1. Wywołaj ask CLI (feedback 5)
    run_cli_command(["ask", "Test pytanie dla Agent Zero"])
    # 2. Wywołaj code CLI (feedback 5)
    run_cli_command(["code", "Stwórz testową funkcję sortowania"])
    # 3. Poczekaj krótko by baza miała czas sie zaktualizować
    time.sleep(1)
    # 4. Sprawdź czy feedbacky są w DB
    test_feedback_storage()
    # 5. Sprawdź output compare-models
    test_compare_models_output()
    # 6. Test orchestratora
    test_orchestrator_model_selection()
    # 7. Test WebSocket (uruchom oddzielnie backend FastAPI zanim)
    print("Uruchom backend FastAPI na porcie 8000 dla testu WebSocket i potem naciśnij Enter...")
    input()
    asyncio.get_event_loop().run_until_complete(test_websocket_echo())
    print("Wszystkie testy przeszły pomyślnie!")
