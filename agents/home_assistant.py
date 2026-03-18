import asyncio
from datetime import datetime, timedelta
from typing import Literal
import uuid
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from audio_functions.genai_audio_utils import convert_text_to_speech, convert_speech_to_text
from audio_functions.audio_listening import get_user_request
from agents.agents import main_agent, quirky_agent
from agents.utils import async_resilient
from tools.timer_tool import check_timers
from log_config.logging_config import get_conversation_logger, logger

class HomeAssistant():
    def __init__(self, memory_timeframe: int = 5, no_audio: bool = False, quirky_response_interval: Literal[None, 3, 5, 10, 15] = 10, dev: bool = False):
        self.conversation_history = []
        self.memory_timeframe = memory_timeframe
        self.end_memory = datetime.today() + timedelta(minutes=memory_timeframe)
        self.quirky_responses = []
        self.no_audio = no_audio
        self.quirky_response_interval = quirky_response_interval
        self.dev = dev
        self.conversation_logger = get_conversation_logger(dev=dev)
        self.conversation_id = str(uuid.uuid4())
        logger.info(f"HomeAssistant initialized with conversation_id: {self.conversation_id} (dev={dev})")

    async def main(self):
        while True:
            try:
                timers_rang = check_timers()

                if timers_rang:
                    for timer in timers_rang:
                        message = f"Timer {timer} has finished."
                        await self.respond_audio(message, tone='neutral')
                        logger.info(f"Timer alarm: {timer}")

                if self.no_audio:
                    wake_word = "hey_jarvis"
                    user_request_text = input("Type your request: ")
                    logger.info(f"User text request received with wake_word: {wake_word}")
                    logger.debug(f"User text input: {user_request_text}")
                else:
                    user_request, wake_word = self.listen_for_user_request()
                    user_request_text = self.convert_audio_request_to_text(audio_bytes=user_request.read())
                    logger.info(f"User request received with wake_word: {wake_word}")
                    logger.debug(f"User input: {user_request_text}")
                match wake_word:
                    case 'hey_jarvis':
                        await self.get_answer(user_request_text, tone='neutral')
                    case 'models/wake_words/a_yi_la_poot.onnx':
                        await self.get_answer(user_request_text, tone='sassy')
                    case 'models/wake_words/oiy_mate.onnx':
                        await self.get_answer(user_request_text, tone='scottish')
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self.conversation_logger.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"conversation_id": self.conversation_id}
                )
        

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
                else:
                    print(self.quirky_responses[-1])
                increment += 1

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
        
        # Log the conversation
        response_text = main_agent_answer.result()[-1].content[0]['text']
        self.conversation_logger.log_conversation(
            user_input=message,
            assistant_response=response_text,
            wake_word="",  # Wake word already logged in main()
            tone=tone,
            conversation_id=self.conversation_id,
        )
        logger.info(f"Conversation logged for tone: {tone}")
        
        if not self.no_audio:
            await self.respond_audio(response_text, tone=tone)
        else:
            print(response_text)

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