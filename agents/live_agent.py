import asyncio
from contextlib import asynccontextmanager
import os
import time
import pyaudio
from websockets import ConnectionClosedOK
import yaml
from datetime import date
from typing import Literal
import aioconsole

from langchain_core.messages import (
    HumanMessage,
)
from google import genai
from google.genai import types

from audio_functions.audio_listening import _get_input_device_index
from tools.utils import ADDRESSES
from agents.agents import route_agent, weather_agent, online_search_agent, code_agent, timer_agent
from audio_functions.genai_audio_utils import ignore_alsa_warnings, setup_audio_stream, FORMAT, CHANNELS, RATE, CHUNK
from log_config.logging_config import get_conversation_logger, logger

AI_API_KEY = os.getenv('GOOGLE_AI_KEY')
MAPS_API_KEY = os.getenv('GOOGLE_MAPS_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

_conversation_logger = get_conversation_logger()

AGENT_TOOLS_MAP = {
    "route_agent_tool": route_agent,
    "weather_agent_tool": weather_agent,
    "online_search_agent_tool": online_search_agent,
    "code_agent_tool": code_agent,
    "timer_agent_tool": timer_agent,
}

# Tool definitions for Gemini Live API
GEMINI_TOOL_DEFINITIONS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="route_agent_tool",
                description="An agent capable of performing Google maps queries to get the route to certain places.",
                parameters={
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The user's query about routes, navigation, traffic, or directions"
                        }
                    },
                    "required": ["input"]
                }
            ),
            types.FunctionDeclaration(
                name="weather_agent_tool",
                description="An agent capable of performing weather queries to get the current, daily, and hourly weather.",
                parameters={
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The user's query about current, daily, or hourly weather"
                        }
                    },
                    "required": ["input"]
                }
            ),
            types.FunctionDeclaration(
                name="online_search_agent_tool",
                description="An agent capable of performing online searches to get accurate, relevant and timely information.",
                parameters={
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The user's query for information search"
                        }
                    },
                    "required": ["input"]
                }
            ),
            types.FunctionDeclaration(
                name="code_agent_tool",
                description="An agent capable of answering queries that are best answered by code execution.",
                parameters={
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The user's query that requires code execution to answer"
                        }
                    },
                    "required": ["input"]
                }
            ),
            types.FunctionDeclaration(
                name="timer_agent_tool",
                description="An agent capable of setting timers, listing timers and deleting timers.",
                parameters={
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The user's query about managing timers"
                        }
                    },
                    "required": ["input"]
                }
            ),
            types.FunctionDeclaration(
                name="end_chat_tool",
            ),
        ]
    )
]

class EndSession(Exception):
    """Custom exception to trigger shutdown"""
    pass

@asynccontextmanager
async def _live_main_agent_setup(tone: Literal["neutral", "sassy", "scottish", "jarjar"] = "neutral", response_mode: Literal["TEXT", "AUDIO"] = "TEXT"):
    with open('agents/agents_config.yaml', 'r') as file:
        live_agent_config = yaml.safe_load(file)['live_agent']


    client = genai.Client(api_key=AI_API_KEY)
    model = live_agent_config['llm_config']['model_name']

    config = types.LiveConnectConfig(
        system_instruction=live_agent_config['llm_config'][tone][f'system_message'].format(
                                    str([name for name, tool in AGENT_TOOLS_MAP.items()]),
                                    date.today().strftime('%Y-%m-%d'),
                                    str(list(ADDRESSES.keys()))),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=live_agent_config['llm_config'][tone]['voice_name']
                )
            )
        ),
        temperature=live_agent_config['llm_config']['temperature'],
        response_modalities=["AUDIO"],
        tools=GEMINI_TOOL_DEFINITIONS,
        input_audio_transcription={},
        output_audio_transcription={},
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
    )

    async with client.aio.live.connect(model=model, config=config) as session:
        yield session, model

async def receive_responses(session, output_stream, shared_state, tone, model_name):
    text_input = ""
    text_output = ""
    input_tokens_text = 0
    input_tokens_audio = 0
    output_tokens_audio = 0
    thinking_tokens = 0

    try:
        while True:
            async for response in session.receive():
                if response and response.usage_metadata and response.usage_metadata.prompt_token_count:
                    input_tokens_text += response.usage_metadata.prompt_tokens_details[0].token_count
                    input_tokens_audio += response.usage_metadata.prompt_tokens_details[1].token_count
                if response and response.usage_metadata and response.usage_metadata.response_token_count:
                    output_tokens_audio += response.usage_metadata.response_token_count
                    thinking_tokens += response.usage_metadata.thoughts_token_count if response.usage_metadata.thoughts_token_count else 0

                if response and response.server_content and response.server_content.output_transcription:
                    text_output += response.server_content.output_transcription.text
                if response and response.server_content and response.server_content.input_transcription:
                    text_input += response.server_content.input_transcription.text

                # Handle tool calls first
                if response and response.tool_call:
                    function_responses = []
                    
                    for function_call in response.tool_call.function_calls:
                        if function_call.name == "end_chat_tool":
                            logger.info("End chat tool called. Ending session.")
                            await session.close()
                            raise EndSession()
                        
                        agent = AGENT_TOOLS_MAP.get(function_call.name)
                        
                        if agent:
                            tool_input = function_call.args.get("input")
                            result = agent.invoke(input = [HumanMessage(content=tool_input)])
                            
                            function_responses.append(
                                types.FunctionResponse(
                                    name=function_call.name,
                                    id=function_call.id,
                                    response={"result": str(result[-1].content[-1]['text'])}
                                )
                            )

                    # 5. Send results back so Gemini can continue speaking
                    await session.send_tool_response(function_responses=function_responses)
                
                # Then handle the generated response (text or audio)
                if response and response.server_content and response.server_content.model_turn:
                    for part in response.server_content.model_turn.parts:
                        if part.inline_data:
                            output_stream.write(part.inline_data.data)
                
                # Print completion info when turn is complete
                if response and response.server_content and response.server_content.turn_complete:
                    print("Turn complete.")
                    print(f"Agent text response : {text_output}")

                    _conversation_logger.log_conversation(text_input, text_output, None, tone) 
                    _conversation_logger.log_token_usage('Live Agent Text', input_tokens_text, 0, thinking_tokens, input_tokens_text + thinking_tokens, model_name)
                    _conversation_logger.log_token_usage('Live Agent Audio', input_tokens_audio, output_tokens_audio, 0, input_tokens_audio + output_tokens_audio, model_name)
                    shared_state['last_ping'] = time.time()  # Update last ping time at the end of the turn
                    # Reset for next turn
                    text_input = ""
                    text_output = ""
    except ConnectionClosedOK as e:
        # Handle session closure (ConnectionClosedOK or other websocket errors)
        logger.debug(f"receive_responses ended: {e}")

async def stream_mic(session, input_stream):
    print("Microphone active. Talk whenever you're ready.")
    while True: # Keep the mic stream running indefinitely
        try:
            # Read a chunk of audio
            input_byte = input_stream.read(CHUNK, exception_on_overflow=False)
            # Send it to the session
            logger.debug(f"Sending audio chunk of size {len(input_byte)}")
            await session.send_realtime_input(audio=types.Blob(data=input_byte, mime_type=f"audio/pcm;rate={RATE}")) 
            
            # Tiny sleep to yield control to the event loop
            await asyncio.sleep(0) 
        except Exception as e:
            print(f"Mic Error: {e}")
            break

async def listen_text(session):
    print("Console chat active. Type your message and press Enter.")    
    while True:
        try:
            text_input = await aioconsole.ainput("User : ")
            
            if text_input and text_input.strip():
                # Send the text to the Gemini session
                logger.info(f"Sending text input to Gemini: {text_input}")
                _conversation_logger.log_conversation(text_input, "", "text_input", "")
                await session.send_realtime_input(text=text_input)
        except asyncio.CancelledError:
            # Task was cancelled (session ended)
            break
    logger.info("Text listener ending due to session inactivity.")

async def limit_time(session, session_time_limit_seconds, shared_state):
    while True:
        await asyncio.sleep(int(session_time_limit_seconds/10))  # Check more frequently than the limit
        exit_time = shared_state['last_ping'] + session_time_limit_seconds
        if time.time() >= exit_time:
            logger.info("Session inactive time limit reached. Closing session.")
            await session.close()
            raise EndSession()


async def live_main_agent(tone: Literal["neutral", "sassy", "scottish", "jarjar"] = "neutral", 
                          response_mode: Literal["TEXT", "AUDIO"] = "TEXT", 
                          session_time_limit_seconds: int = 300):
    try:
        async with _live_main_agent_setup(tone=tone, response_mode=response_mode) as (session, model_name):
            output_stream, p = setup_audio_stream(FORMAT, CHANNELS, RATE)

            with ignore_alsa_warnings():
                pa = pyaudio.PyAudio()

            input_stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE,
                            input=True, frames_per_buffer=CHUNK, input_device_index=_get_input_device_index(pa))
            
            shared_state = {"last_ping": time.time()}

            # Send initial greeting and signal end of input
            await session.send_realtime_input(text="Hello !")
            async with asyncio.TaskGroup() as tg:
                tg.create_task(receive_responses(session, output_stream, shared_state, tone, model_name))
                if response_mode == "AUDIO":
                    tg.create_task(stream_mic(session, input_stream))
                elif response_mode == "TEXT":
                    tg.create_task(listen_text(session))
                tg.create_task(limit_time(session, session_time_limit_seconds, shared_state))
    except* EndSession:
        logger.info("Session ended.")
    finally:
        logger.info("Cleaning up audio streams.")
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()
        input_stream.stop_stream()
        input_stream.close()
