import os
import yaml
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import add_messages
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    BaseMessage,
    ToolCall,
)
from langgraph.func import entrypoint, task

from tools.places_tools import get_route, ADDRESSES

ai_api_key = os.getenv('GOOGLE_AI_KEY')

with open('agents/agent_config.yml', 'r') as file:
    config = yaml.safe_load(file)

class route_agent():
    route_agent_config = config.route_agent
    llm = ChatGoogleGenerativeAI(google_api_key=ai_api_key, 
                                    model=route_agent_config.llm_config.model_name,
                                    temperature=route_agent_config.llm_cionfig.temperature)
    tools = [get_route]
    
    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = route_agent_config.prompt_config.system_prompt.format(ADDRESSES.keys)

    @task
    def call_llm(cls, messages: list[BaseMessage]):
        """LLM decides whether to call a tool or not"""
        return cls.llm_with_tools.invoke(
            [SystemMessage(content=cls.system_prompt)] + messages
        )
    
    @task
    def call_tool(cls, tool_call: ToolCall):
        """Performs the tool call"""
        tool = cls.tools_by_name[tool_call["name"]]
        return tool.invoke(tool_call)

    @entrypoint()
    def route_agent(cls, messages: list[BaseMessage]):
        llm_response = cls.call_llm(messages).result()

        while True:
            if not llm_response.tool_calls:
                break

            # Execute tools
            tool_result_futures = [
                cls.call_tool(tool_call) for tool_call in llm_response.tool_calls
            ]
            tool_results = [fut.result() for fut in tool_result_futures]
            messages = add_messages(messages, [llm_response, *tool_results])
            llm_response = cls.call_llm(messages).result()

        messages = add_messages(messages, llm_response)
        return messages