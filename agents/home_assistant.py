from datetime import datetime, timedelta
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from agents.agents import main_agent

class HomeAssistant():
    def __init__(self, memory_timeframe: int = 5):
        self.conversation_history = []
        self.memory_timeframe = memory_timeframe
        self.end_memory = datetime.today() + timedelta(minutes=memory_timeframe)

    async def get_answer(self, message: str):
        if datetime.today() > self.end_memory:
            self.clear_conversation_history()

        #TODO: Make async to add message that it still waiting, with timeout
        self.add_human_message(message)
        response = main_agent.invoke(self.conversation_history)
        
        self.add_ai_message(response[-1].content)

        self.update_end_memory()
        return response[-1].content[0]['text']

    def update_end_memory(self):
        self.end_memory = datetime.today() + timedelta(minutes=self.memory_timeframe)

    def add_human_message(self, message: str):
        self.conversation_history.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.conversation_history.append(AIMessage(content=message))

    def get_conversation_history(self):
        return self.conversation_history
    
    def clear_conversation_history(self):
        self.conversation_history = []