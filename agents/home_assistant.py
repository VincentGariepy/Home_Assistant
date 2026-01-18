import asyncio
from datetime import datetime, timedelta
from typing import Literal
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from agents.audio_agent import convert_text_to_speech
from agents.agents import main_agent, quirky_agent

class HomeAssistant():
    def __init__(self, memory_timeframe: int = 5, quirky_response_time: int = 10):
        self.conversation_history = []
        self.memory_timeframe = memory_timeframe
        self.end_memory = datetime.today() + timedelta(minutes=memory_timeframe)
        self.quirky_response_time = quirky_response_time
        self.quirky_responses = []

    async def get_answer(self, message: str, tone: Literal['neutral', 'sassy', 'scottish'] = 'neutral', language: str ='English'):
        increment = 1
        if datetime.today() > self.end_memory:
            self.clear_conversation_history()
        
        self.add_human_message(message)

        main_agent_answer = asyncio.create_task(
            asyncio.to_thread(main_agent.invoke, input = {"messages": self.conversation_history, "tone": tone, "language": language}))
        
        while main_agent_answer.done() is False:
            self.quirky_responses.append(quirky_agent.invoke(input = {"messages": [self.conversation_history[-1].content] + self.quirky_responses, "tone": tone, "language": language})[-1].content)
            self.respond_audio(self.quirky_responses[-1], file_name=f'response_{increment}.wav', tone=tone)
            increment += 1
            yield self.quirky_responses[-1]

            _, _ = await asyncio.wait(
                [main_agent_answer],
                timeout=self.quirky_response_time,
            )        

        await asyncio.sleep(3)  # small delay to ensure proper time to answer last quirky response before main answer
        self.add_ai_message(main_agent_answer.result()[-1].content)

        self.update_end_memory()
        self.respond_audio(main_agent_answer.result()[-1].content[0]['text'], file_name='final_response.wav', tone=tone)
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

    def respond_audio(self, message: str, file_name, tone: str):
        convert_text_to_speech(message, file_name, tone=tone)