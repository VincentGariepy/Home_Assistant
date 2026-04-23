from google import genai
from google.genai import types
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
import os
import numpy as np
import yaml
from log_config.logging_config import get_conversation_logger, logger
import asyncio
import websockets.exceptions

_conversation_logger = get_conversation_logger()

AI_API_KEY = os.getenv('GOOGLE_AI_KEY')
with open('agents/agents_config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

import pyaudio
from google import genai
from contextlib import contextmanager

# Gemini Audio Configuration
# 24kHz, 16-bit PCM, Mono is standard for Gemini Live
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000 
CHUNK = 512

# WebSocket configuration for keepalive and timeout
WS_PING_INTERVAL = 20  # Send ping every 20 seconds
WS_PING_TIMEOUT = 10   # Wait 10 seconds for pong response

GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
GOOGLE_MAPS_KEY = os.getenv('GOOGLE_MAPS_KEY')

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
    logger.debug("Searching for audio output device...")
    target_index = None
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        # Filter for output devices only
        if info['maxOutputChannels'] > 0:
            # Auto-select a USB device or specific HAT if found
            # Change "USB" to "Headphones" or your specific device name
            if "USB" in info['name']: 
                logger.debug(f"Found USB audio device at index {i}: {info['name']}")
                target_index = i

    # If we found a specific USB device, use it. Otherwise, let PyAudio choose default.
    if target_index is not None:
        logger.info(f"Using audio output device at index {target_index}")
    else:
        logger.debug("No specific USB device found, using default output device")
    return target_index

def _get_text_to_speech_config(content:str, tone: str):
    logger.debug(f"Configuring text-to-speech with tone: {tone}")
    agent_config = CONFIG['text_to_speech_agent']['llm_config']

    if tone == 'sassy':
        llm_config = agent_config['sassy']
    elif tone == 'scottish':
        llm_config = agent_config['scottish']
    else:
        llm_config = agent_config['neutral']  

    logger.debug(f"Using voice: {llm_config['voice_name']}")
    content = llm_config['system_message'].format(content)

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=llm_config['voice_name']
                )
            )
        )
    )
    return content, agent_config['model_name'], config

async def _receive_with_timeout(session):
    """
    Helper async generator that wraps session.receive() with better error handling.
    Yields responses from the session until the stream is exhausted.
    """
    try:
        async for response in session.receive():
            yield response
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"WebSocket connection closed during receive: {e}")
        raise
    except Exception as e:
        logger.error(f"Error receiving from session: {e}")
        raise

def setup_audio_stream(format, channels, rate):
    logger.info(f"Setting up audio stream: format={format}, channels={channels}, rate={rate}Hz")
    with ignore_alsa_warnings():
        p = pyaudio.PyAudio()
    
    output_device_index = _get_output_device_index(p)
    try:
        stream = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        output=True,
                        output_device_index=output_device_index)
        logger.info("Audio stream opened successfully")
        return stream, p
    except OSError as e:
        logger.error(f"Error opening audio device: {e}")
        logger.error("Try changing the sample rate to 44100 or 48000 if your hardware doesn't support 24000.")
        raise

async def convert_text_to_speech(content: str, tone: str):
    logger.info(f"Converting text to speech with tone: {tone}")
    stream, p = setup_audio_stream(FORMAT, CHANNELS, RATE)
    client = genai.Client(api_key=AI_API_KEY)
    content, model_id, config = _get_text_to_speech_config(content=content, tone=tone)

    try:
        logger.debug(f"Connecting to model: {model_id}")
        usage_metadata = None
        
        async with client.aio.live.connect(model=model_id, config=config) as session:
            await session.send_realtime_input(text=content)
            
            # Use a timeout for receiving audio to prevent hanging
            try:
                async for response in asyncio.wait_for(_receive_with_timeout(session), timeout=120):
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data:
                                stream.write(part.inline_data.data)
                    if response.usage_metadata and response.usage_metadata.prompt_token_count:
                        usage_metadata = response.usage_metadata
            except asyncio.TimeoutError:
                logger.error(f"Text-to-speech response timed out after 120 seconds")
                raise



        logger.info(f"Text-to-speech completed.")
            
        # Log token usage if available
        try:
            if hasattr(usage_metadata, 'prompt_tokens_details'):
                input_tokens = usage_metadata.prompt_tokens_details[0].token_count
                output_tokens = usage_metadata.prompt_tokens_details[1].token_count + usage_metadata.candidates_token_count
                total_tokens = input_tokens + output_tokens
                
                _conversation_logger.log_token_usage(
                    agent_name='text_to_speech_agent',
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=0,
                    total_tokens=total_tokens,
                    model_name=model_id,
                )
                logger.debug(f"Token usage for text-to-speech: input={input_tokens}, output={output_tokens}, total={total_tokens}")
        except Exception as e:
            logger.debug(f"Could not extract token usage from TTS response: {e}")
    except Exception as e:
        logger.error(f"Error during text-to-speech conversion: {e}", exc_info=True)
        raise
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        logger.debug("Audio stream closed")