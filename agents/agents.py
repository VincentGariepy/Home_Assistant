import os
import yaml
from datetime import date
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import add_messages
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    BaseMessage,
    ToolCall,
)
from langgraph.func import entrypoint, task

from tools.utils import ADDRESSES
from tools.places_tools import get_route
from tools.weather_tools import get_current_weather, get_daily_weather, get_hourly_weather
from agents.agent_init_data import AgentInitData

ai_api_key = os.getenv('GOOGLE_AI_KEY')

with open('agents/agents_config.yml', 'r') as file:
    config = yaml.safe_load(file)

@task
def _call_llm(messages: list[BaseMessage], agent_data: AgentInitData):
    """LLM decides whether to call a tool or not"""
    print(agent_data.system_prompt)
    return agent_data.llm_with_tools.invoke(
        [SystemMessage(content=agent_data.system_prompt)] + messages
    )

@task
def _call_tool(tool_call: ToolCall, agent_data: AgentInitData):
    """Performs the tool call"""
    tool = agent_data.tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)

def _init_route_agent():
    route_agent_config = config['route_agent']
    llm = ChatGoogleGenerativeAI(google_api_key=ai_api_key, 
                                model=route_agent_config['llm_config']['model_name'],
                                temperature=route_agent_config['llm_config']['temperature'])
    tools = [get_route]

    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)


    system_prompt = route_agent_config['prompt_config']['system_message'].format(str(list(ADDRESSES.keys())), date.today().strftime())
    return AgentInitData(tools_by_name=tools_by_name, 
                         llm_with_tools=llm_with_tools,
                         system_prompt=system_prompt)

@entrypoint()
def route_agent(messages: list[BaseMessage]):
    route_agent_data = _init_route_agent()

    llm_response = _call_llm(messages, route_agent_data).result()

    while True:
        if not llm_response.tool_calls:
            break

        # Execute tools
        tool_result_futures = [
            _call_tool(tool_call, route_agent_data) for tool_call in llm_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [llm_response, *tool_results])
        llm_response = _call_llm(messages, route_agent_data).result()

    messages = add_messages(messages, llm_response)
    return messages

def _init_weather_agent():
    weather_agent_config = config['weather_agent']
    llm = ChatGoogleGenerativeAI(google_api_key=ai_api_key, 
                                model=weather_agent_config['llm_config']['model_name'],
                                temperature=weather_agent_config['llm_config']['temperature'])
    tools = [get_current_weather, get_daily_weather, get_hourly_weather]

    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)


    system_prompt = weather_agent_config['prompt_config']['system_message'].format(
                                    str(list(ADDRESSES.keys())), 
                                    date.today().strftime('%Y-%m-%d'))

    return AgentInitData(tools_by_name=tools_by_name, 
                         llm_with_tools=llm_with_tools,
                         system_prompt=system_prompt)

@entrypoint()
def weather_agent(messages: list[BaseMessage]):
    weather_agent_data = _init_weather_agent()

    llm_response = _call_llm(messages, weather_agent_data).result()

    while True:
        if not llm_response.tool_calls:
            break

        # Execute tools
        tool_result_futures = [
            _call_tool(tool_call, weather_agent_data) for tool_call in llm_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [llm_response, *tool_results])
        llm_response = _call_llm(messages, weather_agent_data).result()

    messages = add_messages(messages, llm_response)
    return messages

def _init_online_search_agent():
    online_search_agent_config = config['online_search_agent']
    llm = ChatGoogleGenerativeAI(google_api_key=ai_api_key, 
                                model=online_search_agent_config['llm_config']['model_name'],
                                temperature=online_search_agent_config['llm_config']['temperature'])
    tools = [get_current_weather, get_daily_weather, get_hourly_weather]

    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)


    system_prompt = online_search_agent_config['prompt_config']['system_message']

    return AgentInitData(tools_by_name=tools_by_name, 
                         llm_with_tools=llm_with_tools,
                         system_prompt=system_prompt)