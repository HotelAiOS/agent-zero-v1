"""Test RabbitMQ connection"""
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from messaging import MessageBus, BusConfig

print("🧪 Test połączenia z RabbitMQ...\n")

# Stwórz bus
config = BusConfig(host="localhost", port=5672)
bus = MessageBus(config)

# Połącz
if bus.connect():
    print("✅ Połączono z RabbitMQ!")
    
    # Zadeklaruj exchange
    if bus.declare_exchange("test_exchange", "topic"):
        print("✅ Exchange zadeklarowany")
    
    # Zadeklaruj kolejkę
    if bus.declare_queue("test_queue"):
        print("✅ Kolejka zadeklarowana")
    
    # Zbinduj
    if bus.bind_queue("test_queue", "test_exchange", "test.#"):
        print("✅ Binding utworzony")
    
    bus.disconnect()
    print("\n🎉 Wszystkie testy przeszły!")
else:
    print("❌ Nie można połączyć z RabbitMQ")
