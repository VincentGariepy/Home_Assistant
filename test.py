import asyncio
import os
from google import genai
from google.genai import types
from joblib import logger

client = genai.Client(api_key=os.getenv("GOOGLE_AI_KEY"))
model = "gemini-3.1-flash-live-preview"

config = {
    "response_modalities": ["AUDIO"],
    "output_audio_transcription": {}
}

async def send_text_input(session):
    while True:
        # Run the blocking input() in a thread so it doesn't freeze audio
        text_input = await asyncio.to_thread(input, "User: ")
            
        if text_input.strip():
            # Send the text to the Gemini session
            print(f"Sending text input to Gemini: {text_input}")
            await session.send_realtime_input(text=text_input)

async def respond_to_gemini(session):
    while True:
        async for response in session.receive():
            print(response)
            if response.server_content.model_turn:
                print("Model turn:", response.server_content.model_turn)
            if response.server_content.output_transcription:
                print("Transcript:", response.server_content.output_transcription.text)

async def main():
    async with client.aio.live.connect(model=model, config=config) as session:
        print("Connected to Gemini. Start speaking!")
        async with asyncio.TaskGroup() as tg:
            tg.create_task(send_text_input(session))
            tg.create_task(respond_to_gemini(session))

if __name__ == "__main__":
    asyncio.run(main())