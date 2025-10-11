import asyncio
import websockets

async def test_ws():
    async with websockets.connect("ws://localhost:8002/ws/testsession") as ws:
        await ws.send("Cześć, jestem użytkownikiem!")
        print("Wysłano: Cześć, jestem użytkownikiem!")
        response = await ws.recv()
        print("Odpowiedź:", response)

asyncio.run(test_ws())
