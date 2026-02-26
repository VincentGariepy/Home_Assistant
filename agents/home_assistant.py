import asyncio
from datetime import datetime, timedelta
from typing import Literal
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from audio_functions.genai_audio_utils import convert_text_to_speech, convert_speech_to_text
from audio_functions.audio_listening import get_user_request
from agents.agents import main_agent, quirky_agent
from agents.utils import async_resilient

class HomeAssistant():
    def __init__(self, memory_timeframe: int = 5, no_audio: bool = False, quirky_response_interval: Literal[None, 3, 5, 10, 15] = 10):
        self.conversation_history = []
        self.memory_timeframe = memory_timeframe
        self.end_memory = datetime.today() + timedelta(minutes=memory_timeframe)
        self.quirky_responses = []
        self.no_audio = no_audio
        self.quirky_response_interval = quirky_response_interval

    async def main(self):
        while True:
            user_request, wake_word = self.listen_for_user_request()
            user_request_text = self.convert_audio_request_to_text(audio_bytes=user_request.read())

            match wake_word:
                case 'hey_jarvis':
                    await self.get_answer(user_request_text, tone='neutral', language='English')
                case 'models/wake_words/a_yi_la_poot.onnx':
                    await self.get_answer(user_request_text, tone='sassy', language='English')
                case 'models/wake_words/oiy_mate.onnx':
                    await self.get_answer(user_request_text, tone='scottish', language='English')
        

    @async_resilient(retries=2, timeout=120)
    async def get_answer(self, message: str,
                         tone: Literal['neutral', 'sassy', 'scottish'] = 'neutral'):
        increment = 1
        if datetime.today() > self.end_memory:
            self.clear_conversation_history()
        
        self.add_human_message(message)

        main_agent_answer = asyncio.create_task(
            asyncio.to_thread(main_agent.invoke, input = {"messages": self.conversation_history, "tone": tone}))
        
        if self.quirky_response_interval is not None:
            while main_agent_answer.done() is False:
                self.quirky_responses.append(quirky_agent.invoke(input = {"messages": [self.conversation_history[-1].content] + self.quirky_responses, "tone": tone})[-1].content)
                if not self.no_audio:
                    await self.respond_audio(self.quirky_responses[-1], tone=tone)
                increment += 1
                yield self.quirky_responses[-1]

                _, _ = await asyncio.wait(
                    [main_agent_answer],
                    timeout=self.quirky_response_interval,
                )        

            await asyncio.sleep(3)  # small delay to ensure proper time to answer last quirky response before main answer
        else:
            _, _ = await asyncio.wait(
                        [main_agent_answer],
                        timeout=None,
                    )
        self.add_ai_message(main_agent_answer.result()[-1].content)

        self.update_end_memory()
        if not self.no_audio:
            await self.respond_audio(main_agent_answer.result()[-1].content[0]['text'], tone=tone)
        yield main_agent_answer.result()[-1].content[0]['text']

    def update_end_memory(self):
        self.end_memory = datetime.today() + timedelta(minutes=self.memory_timeframe)

    def add_human_message(self, message: str):
        self.conversation_history.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.conversation_history.append(AIMessage(content=message))

    def get_conversation_history(self) -> list:
        return self.conversation_history
    
    def clear_conversation_history(self):
        self.conversation_history = []
        self.quirky_responses = []

    async def respond_audio(self, message: str, tone: str):
        await convert_text_to_speech(message, tone=tone)

    def convert_audio_request_to_text(self, file_path: str = None, audio_bytes: bytes = None) -> str:
        return convert_speech_to_text(file_path=file_path, audio_bytes=audio_bytes)
    
    def listen_for_user_request(self) -> tuple:
        return get_user_request()