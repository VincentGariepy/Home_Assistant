from google import genai
from google.genai import types
import wave
import os

import yaml

AI_API_KEY = os.getenv('GOOGLE_AI_KEY')
with open('agents/agents_config.yml', 'r') as file:
    CONFIG = yaml.safe_load(file)

import asyncio
import pyaudio
import sys
from google import genai

# Gemini Audio Configuration
# 24kHz, 16-bit PCM, Mono is standard for Gemini Live
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000 
CHUNK = 512

import os
import sys
from contextlib import contextmanager

# Define the Error Silencer Context Manager
@contextmanager
def ignore_alsa_warnings():
    """
    Temporarily redirects C-level stderr to /dev/null to silence 
    ALSA and PortAudio warnings.
    """
    # 1. Open a file descriptor to /dev/null
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        # If we can't open devnull, just run the code without silencing
        yield
        return

    # 2. Save the original stderr file descriptor (usually 2)
    original_stderr = os.dup(2)

    try:
        # 3. Replace stderr (2) with devnull
        os.dup2(devnull, 2)
        yield
    finally:
        # 4. Restore the original stderr
        os.dup2(original_stderr, 2)
        os.close(original_stderr)
        os.close(devnull)

def _get_output_device_index(p):
    """
    Helper to find the correct audio output device.
    Useful for Raspberry Pi where 'default' might be HDMI.
    """
    target_index = None
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        # Filter for output devices only
        if info['maxOutputChannels'] > 0:
            # Auto-select a USB device or specific HAT if found
            # Change "USB" to "Headphones" or your specific device name
            if "USB" in info['name']: 
                target_index = i

    # If we found a specific USB device, use it. Otherwise, let PyAudio choose default.
    return target_index

def _get_text_to_speech_config(content:str, tone: str):
    agent_config = CONFIG['text_to_speech_agent']['llm_config']

    if tone == 'sassy':
        llm_config = agent_config['sassy']
    elif tone == 'scottish':
        llm_config = agent_config['scottish']
    else:
        llm_config = agent_config['neutral']  

    content = llm_config['system_message'].format(content)

    return content, agent_config['model_name'], {'response_modalities': ["AUDIO"],
        'speech_config': {
            'voice_config': {
                'prebuilt_voice_config': {
                    'voice_name': llm_config['voice_name'],
                }
            }
        }
    }

async def convert_text_to_speech(content: str, tone: str):
    with ignore_alsa_warnings():
        p = pyaudio.PyAudio()
    
    # 1. Select Device (Crucial for Pi)
    output_device_index = _get_output_device_index(p)
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True,
                        output_device_index=output_device_index)
    except OSError as e:
        print(f"Error opening audio device: {e}")
        print("Try changing the sample rate to 44100 or 48000 if your hardware doesn't support 24000.")
        return

    client = genai.Client(api_key=AI_API_KEY)
    content, model_id, config = _get_text_to_speech_config(content=content, tone=tone)

    async with client.aio.live.connect(model=model_id, config=config) as session: 
        await session.send(input=content, end_of_turn=True)       
        async for response in session.receive():
            server_content = response.server_content
            if server_content is not None:
                model_turn = server_content.model_turn
                if model_turn is not None:
                    for part in model_turn.parts:
                        if part.inline_data:
                            # Play audio chunk immediately
                            stream.write(part.inline_data.data)

    stream.stop_stream()
    stream.close()
    p.terminate()

def convert_speech_to_text(file_path: str) -> str:
    agent_config = CONFIG['speech_to_text_agent']['llm_config']
    client = genai.Client(api_key=AI_API_KEY)

    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    client = genai.Client()
    response = client.models.generate_content(
    model=agent_config['model_name'],
    contents=[
        agent_config['system_message'],
        types.Part.from_bytes(
        data=audio_bytes,
        mime_type='audio/mp3',
        )
    ]
    )

    print(response.text)