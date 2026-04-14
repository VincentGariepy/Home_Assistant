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
from log_config.log_analyzer import send_monthly_email_report

class HomeAssistant():
    def __init__(self, memory_timeframe: int = 5, no_audio: bool = False, quirky_response_interval: Literal[None, 3, 5, 10, 15] = 10):
        self.conversation_history = []
        self.memory_timeframe = memory_timeframe
        self.end_memory = datetime.today() + timedelta(minutes=memory_timeframe)
        self.quirky_responses = []
        self.no_audio = no_audio
        self.quirky_response_interval = quirky_response_interval
        self.conversation_logger = get_conversation_logger()
        self.conversation_id = str(uuid.uuid4())
        logger.info(f"HomeAssistant initialized with conversation_id: {self.conversation_id}")
        self.conversation_logger.set_conversation_id(self.conversation_id)

    async def _check_timers_periodically(self):
        """Background task that checks timers periodically."""
        while True:
            try:
                await asyncio.sleep(0.1)  # Check every 100ms
                timers_rang = check_timers()

                if timers_rang:
                    for timer in timers_rang:
                        message = f"Timer {timer} has finished."
                        await self.respond_audio(message, tone='neutral')
                        logger.info(f"Timer alarm: {timer}")
            except Exception as e:
                logger.error(f"Error in timer check task: {e}", exc_info=True)

    async def main(self):
        # Start the background timer check task
        timer_task = asyncio.create_task(self._check_timers_periodically())
        
        try:
            while True:
                send_monthly_email_report()  # Check if it's time to send the monthly report
                try:
                    if self.no_audio:
                        wake_word = "hey_jarvis_v0.1"
                        # Run blocking input in a thread so event loop can continue
                        user_request_text = await asyncio.to_thread(input, "Type your request: ")
                        logger.info(f"User text request received with wake_word: {wake_word}")
                        logger.debug(f"User text input: {user_request_text}")
                    else:
                        # Run blocking listen in a thread so event loop can continue
                        user_request, wake_word = await asyncio.to_thread(self.listen_for_user_request)
                        user_request_text = self.convert_audio_request_to_text(audio_bytes=user_request.read())
                        logger.info(f"User request received with wake_word: {wake_word}")
                        logger.debug(f"User input: {user_request_text}")
                    match wake_word:
                        case 'hey_jarvis_v0.1':
                            await self.get_answer(user_request_text, tone='neutral')
                        case 'heille_la_pute' | 'hey_babe':
                            await self.get_answer(user_request_text, tone='sassy')
                        case 'oi_mate':
                            await self.get_answer(user_request_text, tone='scottish')
                        case 'jarjar':
                            await self.get_answer(user_request_text, tone='jarjar')
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    self.conversation_logger.log_error(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        context={"conversation_id": self.conversation_id}
                    )
        finally:
            timer_task.cancel()
        

    @async_resilient(retries=2, timeout=120)
    async def get_answer(self, message: str,
                         tone: Literal['neutral', 'sassy', 'scottish', 'jarjar'] = 'neutral'):
        if datetime.today() > self.end_memory:
            self.clear_conversation_history()
        
        self.add_human_message(message)
        
        match tone:
            case 'neutral':
                await self.respond_audio("Hello, I'm processing your request...", tone=tone)
            case 'sassy':
                await self.respond_audio("Hold on, gimme a sec.", tone=tone)
            case 'scottish':
                await self.respond_audio("Aye, I'm on it.", tone=tone)
            case 'jarjar':
                await self.respond_audio("Meesa thinking, justen one moment please!", tone=tone)

        main_agent_answer = asyncio.create_task(
            asyncio.to_thread(main_agent.invoke, input = {"messages": self.conversation_history, "tone": tone}))
        
        _, _ = await asyncio.wait(
                    [main_agent_answer],
                    timeout=self.quirky_response_interval,
                ) 
        
        if self.quirky_response_interval is not None:
            while main_agent_answer.done() is False:
                self.quirky_responses.append(quirky_agent.invoke(input = {"messages": [AIMessage(content=resp) for resp in self.quirky_responses] + [self.conversation_history[-1]], "tone": tone})[-1].content)
                await self.respond_audio(self.quirky_responses[-1][-1]['text'], tone=tone)

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
        )
        logger.info(f"Conversation logged for tone: {tone}")
        
        await self.respond_audio(response_text, tone=tone)

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
        self.conversation_id = str(uuid.uuid4())
        self.conversation_logger.set_conversation_id(self.conversation_id)
        logger.info(f"Conversation history cleared. New conversation_id: {self.conversation_id}")

    async def respond_audio(self, message: str, tone: str):
        if not self.no_audio:
            await convert_text_to_speech(message, tone=tone)
        else:
            print(message)

    def convert_audio_request_to_text(self, file_path: str = None, audio_bytes: bytes = None) -> str:
        return convert_speech_to_text(file_path=file_path, audio_bytes=audio_bytes)
    
    def listen_for_user_request(self) -> tuple:
        return get_user_request()