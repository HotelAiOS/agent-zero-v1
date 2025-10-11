"""Test messaging module"""
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from messaging import (
    Message, 
    MessageType, 
    MessagePriority,
    create_task_request
)

print("✅ Import modułu messaging: OK")

# Test tworzenia wiadomości
msg = create_task_request(
    sender_id="backend_1",
    recipient_id="database_1",
    task_description="Create schema for users",
    task_id="task_123",
    project_id="proj_001"
)

print(f"✅ Utworzono message: {msg.message_type.value}")
print(f"   {msg.sender_id} → {msg.recipient_id}")
print(f"   Priorytet: {msg.priority.value}")

# Test serializacji
data = msg.to_dict()
msg2 = Message.from_dict(data)

print("✅ Serializacja/deserializacja: OK")
print("\n🎉 Wszystkie testy przeszły!")
