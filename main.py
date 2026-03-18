import asyncio
from agents.home_assistant import HomeAssistant


# To create an executable, run: `pyinstaller --onedir main.py``
if __name__ == "__main__":
    assistant = HomeAssistant(memory_timeframe=5, no_audio=True, dev=True)

    asyncio.run(assistant.main())