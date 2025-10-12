"""
Test Agent-to-Agent Communication via RabbitMQ
Dwóch agentów komunikuje się przez message bus
"""

import sys
import time
import threading
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

from messaging import (
    AgentCommunicator,
    Message,
    MessageType,
    MessagePriority
)

print("\n" + "="*70)
print("🧪 TEST: Agent-to-Agent Communication")
print("="*70 + "\n")

# === AGENT BACKEND ===

def backend_agent():
    """Agent Backend - wysyła zadanie do Database"""
    print("🔵 Backend Agent: Inicjalizacja...")
    
    comm = AgentCommunicator("backend_1", auto_connect=True)
    
    def handle_response(msg: Message):
        """Obsługa odpowiedzi od Database"""
        if msg.message_type == MessageType.TASK_RESPONSE:
            print(f"\n🔵 Backend otrzymał odpowiedź od {msg.sender_id}:")
            print(f"   Temat: {msg.subject}")
            print(f"   Treść: {msg.content}")
            print(f"   Payload: {msg.payload}")
    
    # Zarejestruj handler
    comm.on_message(handle_response)
    
    # Rozpocznij nasłuchiwanie w tle
    comm.start_listening(block=False)
    
    print("🔵 Backend Agent: Gotowy\n")
    
    # Poczekaj 2s żeby Database był gotowy
    time.sleep(2)
    
    # Wyślij zadanie do Database
    print("🔵 Backend → Database: Wysyłam zadanie...")
    success = comm.request_task(
        recipient_id="database_1",
        task_description="Create schema for 'users' table with fields: id, email, password, created_at",
        task_id="task_001",
        project_id="proj_test",
        priority=MessagePriority.HIGH,
        payload={
            "table": "users",
            "fields": ["id", "email", "password", "created_at"]
        }
    )
    
    if success:
        print("   ✅ Zadanie wysłane\n")
    else:
        print("   ❌ Błąd wysyłania\n")
    
    # Poczekaj na odpowiedź
    time.sleep(5)
    
    # Zatrzymaj
    comm.stop_listening()
    comm.disconnect()
    print("🔵 Backend Agent: Zakończono")


# === AGENT DATABASE ===

def database_agent():
    """Agent Database - odbiera zadanie, przetwarza, odpowiada"""
    print("🟢 Database Agent: Inicjalizacja...")
    
    comm = AgentCommunicator("database_1", auto_connect=True)
    
    def handle_task(msg: Message):
        """Obsługa zadania od Backend"""
        if msg.message_type == MessageType.TASK_REQUEST:
            print(f"\n🟢 Database otrzymał zadanie od {msg.sender_id}:")
            print(f"   Task ID: {msg.task_id}")
            print(f"   Treść: {msg.content}")
            print(f"   Payload: {msg.payload}")
            
            # Symulacja przetwarzania
            print("🟢 Database: Przetwarzam... (2s)")
            time.sleep(2)
            
            # Wygeneruj odpowiedź
            schema = f"""
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
            
            # Wyślij odpowiedź
            print("🟢 Database → Backend: Wysyłam odpowiedź...\n")
            comm.send_direct(
                recipient_id=msg.sender_id,
                subject=f"Task {msg.task_id} completed",
                content="Schema created successfully",
                message_type=MessageType.TASK_RESPONSE,
                priority=MessagePriority.NORMAL,
                payload={
                    "schema": schema.strip(),
                    "status": "completed"
                }
            )
    
    # Zarejestruj handler
    comm.on_message(handle_task)
    
    print("🟢 Database Agent: Gotowy, nasłuchuję...\n")
    
    # Blokuj i nasłuchuj (main thread)
    comm.start_listening(block=True)


# === MAIN TEST ===

if __name__ == '__main__':
    try:
        # Uruchom Database w osobnym wątku
        db_thread = threading.Thread(target=database_agent, daemon=True)
        db_thread.start()
        
        # Uruchom Backend w main thread
        backend_agent()
        
        print("\n" + "="*70)
        print("✅ TEST ZAKOŃCZONY SUKCESEM!")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test przerwany (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Błąd testu: {e}")
        import traceback
        traceback.print_exc()
