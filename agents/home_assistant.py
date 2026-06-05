import asyncio
from datetime import datetime, timedelta
from typing import Literal
import uuid

import aioconsole

from audio_functions.genai_audio_utils import convert_text_to_speech
from audio_functions.audio_listening import get_wake_word
from agents.live_agent import live_main_agent
from tools.timer_tool import check_timers
from log_config.logging_config import get_conversation_logger, logger
from log_config.log_analyzer import send_monthly_email_report

WAKE_WORD_MAPPING = {
    'hey_jarvis_v0.1': 'neutral',
    'heille_la_pute': 'sassy',
    'hey_babe': 'sassy',
    'oi_mate': 'scottish',
    'jarjar': 'jarjar'
}

class HomeAssistant():
    def __init__(self, memory_timeframe: int = 300, no_audio: bool = False):
        self.memory_timeframe = memory_timeframe
        self.no_audio = no_audio
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
                #wait 2 seconds before listening for wake word again to avoid multiple triggers in a row
                await asyncio.sleep(2)
                try:
                    response_mode = "AUDIO" if not self.no_audio else "TEXT"
                    if self.no_audio:
                        wake_word = "hey_jarvis_v0.1"
                        # Run blocking input in a thread so event loop can continue
                        user_input = await aioconsole.ainput("Press Enter to chat (or 'q' to quit): ")
                        if user_input.strip().lower() == 'q':
                            logger.info("Exiting Home Assistant.")
                            break
                        logger.info(f"User text request received with wake_word: {wake_word}")
                    else:
                        # Run blocking listen in a thread so event loop can continue
                        wake_word = await asyncio.to_thread(self.listen_for_wake_word)
                        logger.info(f"User request received with wake_word: {wake_word}")

                    await live_main_agent(tone=WAKE_WORD_MAPPING.get(wake_word, 'neutral'), 
                                          response_mode=response_mode, 
                                          session_time_limit_seconds=self.memory_timeframe)
                
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    self.conversation_logger.log_error(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        context={"conversation_id": self.conversation_id}
                    )
        finally:
            timer_task.cancel()  

    async def respond_audio(self, message: str, tone: str):
        if not self.no_audio:
            await convert_text_to_speech(message, tone=tone)
        else:
            print(message)
    
    def listen_for_wake_word(self) -> str:
        wake_word = get_wake_word()
        return wake_word