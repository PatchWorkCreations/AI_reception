import asyncio, websockets

async def main():
    uri = "wss://colory-cycadlike-fransisca.ngrok-free.dev/ws/twilio"
    try:
        async with websockets.connect(uri, subprotocols=["audio.twilio.com"]) as ws:
            print("client ▶ connected to bot")
            await asyncio.sleep(1)
    except Exception as e:
        print("client ▶ FAILED:", e)

if __name__ == "__main__":
    asyncio.run(main())
