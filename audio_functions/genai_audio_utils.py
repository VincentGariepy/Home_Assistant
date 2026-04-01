from google import genai
from google.genai import types
import wave
import os
import logging

import yaml
from log_config.logging_config import ConversationLogger

logger = logging.getLogger(__name__)
_conversation_logger = ConversationLogger()

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

    return content, agent_config['model_name'], {'response_modalities': ["AUDIO"],
        'speech_config': {
            'voice_config': {
                'prebuilt_voice_config': {
                    'voice_name': llm_config['voice_name'],
                }
            }
        }
    }

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
        async with client.aio.live.connect(model=model_id, config=config) as session: 
            await session.send(input=content, end_of_turn=True)
            logger.debug("Text sent to model, waiting for audio response...")
            audio_chunks = 0
            async for response in session.receive():
                server_content = response.server_content
                if server_content is not None:
                    model_turn = server_content.model_turn
                    if model_turn is not None:
                        for part in model_turn.parts:
                            if part.inline_data:
                                # Play audio chunk immediately
                                stream.write(part.inline_data.data)
                                audio_chunks += 1
            logger.info(f"Text-to-speech completed. Played {audio_chunks} audio chunks")
            
            # Log token usage if available
            try:
                if hasattr(session, 'response_metadata') and hasattr(session, 'usage_metadata'):
                    usage = session.usage_metadata
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
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

def convert_speech_to_text(file_path: str = None, audio_bytes: bytes = None) -> str:
    logger.info(f"Converting speech to text from file_path: {file_path}")
    agent_config = CONFIG['speech_to_text_agent']['llm_config']
    client = genai.Client(api_key=AI_API_KEY)

    if file_path is not None:
        logger.debug(f"Loading audio file: {file_path}")
        with open(file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        logger.debug(f"Loaded {len(audio_bytes)} bytes from audio file")

    client = genai.Client()
    try:
        logger.debug(f"Sending audio to model: {agent_config['model_name']}")
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
        logger.info(f"Speech-to-text conversion completed")
        logger.debug(f"Transcribed text: {response.text[:100]}...")
        
        # Log token usage if available
        try:
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                total_tokens = input_tokens + output_tokens
                
                _conversation_logger.log_token_usage(
                    agent_name='speech_to_text_agent',
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=0,
                    total_tokens=total_tokens,
                    model_name=agent_config['model_name'],
                )
                logger.debug(f"Token usage for speech-to-text: input={input_tokens}, output={output_tokens}, total={total_tokens}")
        except Exception as e:
            logger.debug(f"Could not extract token usage from STT response: {e}")
        
        print(response.text)
        return response.text
    except Exception as e:
        logger.error(f"Error during speech-to-text conversion: {e}", exc_info=True)
        raise