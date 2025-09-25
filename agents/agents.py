import os
import yaml
from datetime import date

from google.genai import types

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import add_messages
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    BaseMessage,
    ToolCall,
)
from langgraph.func import entrypoint, task
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

from tools.utils import ADDRESSES
from tools.places_tools import get_route
from tools.weather_tools import get_current_weather, get_daily_weather, get_hourly_weather
from agents.agent_init_data import AgentInitData

ai_api_key = os.getenv('GOOGLE_AI_KEY')
maps_api_key = os.getenv('GOOGLE_MAPS_KEY')
google_cse_id = os.getenv('GOOGLE_CSE_ID')

with open('agents/agents_config.yml', 'r') as file:
    config = yaml.safe_load(file)

@task
def _call_llm(messages: list[BaseMessage], agent_data: AgentInitData):
    """LLM decides whether to call a tool or not"""
    return agent_data.llm_with_tools.invoke(
        [SystemMessage(content=agent_data.system_prompt)] + messages
    )

@task
def _call_tool(tool_call: ToolCall, agent_data: AgentInitData):
    """Performs the tool call"""
    tool = agent_data.tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)

def _tool_calling_agent(agent_data: AgentInitData, messages: list[BaseMessage]):
    llm_response = _call_llm(messages, agent_data).result()

    while True:
        if not llm_response.tool_calls:
            break

        # Execute tools
        tool_result_futures = [
            _call_tool(tool_call, agent_data) for tool_call in llm_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [llm_response, *tool_results])
        llm_response = _call_llm(messages, agent_data).result()

    messages = add_messages(messages, llm_response)
    return messages

def _init_route_agent():
    route_agent_config = config['route_agent']
    llm = ChatGoogleGenerativeAI(google_api_key=ai_api_key, 
                                model=route_agent_config['llm_config']['model_name'],
                                temperature=route_agent_config['llm_config']['temperature'])
    tools = [get_route]

    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)


    system_prompt = route_agent_config['prompt_config']['system_message'].format(str(list(ADDRESSES.keys())))
    return AgentInitData(tools_by_name=tools_by_name, 
                         llm_with_tools=llm_with_tools,
                         system_prompt=system_prompt)

@entrypoint()
def route_agent(messages: list[BaseMessage]):
    route_agent_data = _init_route_agent()

    return _tool_calling_agent(route_agent_data, messages)
   

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

    return _tool_calling_agent(weather_agent_data, messages)

def _init_online_search_agent():
    online_search_agent_config = config['online_search_agent']
    llm = ChatGoogleGenerativeAI(google_api_key=ai_api_key, 
                                model=online_search_agent_config['llm_config']['model_name'],
                                temperature=online_search_agent_config['llm_config']['temperature'])
    
    search = GoogleSearchAPIWrapper()
    search.google_api_key = maps_api_key
    search.google_cse_id = google_cse_id

    grounding_tool = Tool(
        name="google_search",
        description="Search Google for recent results.",
        func=search.run,
    )
    
    tools = [grounding_tool]

    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)


    system_prompt = online_search_agent_config['prompt_config']['system_message']

    return AgentInitData(tools_by_name=tools_by_name, 
                         llm_with_tools=llm_with_tools,
                         system_prompt=system_prompt)

#TODO: Create requriements.txt
@entrypoint()
def online_search_agent(messages: list[BaseMessage]):
    weather_agent_data = _init_online_search_agent()

    return _tool_calling_agent(weather_agent_data, messages)

def _init_code_agent():
    online_search_agent_config = config['code_agent']
    llm = ChatGoogleGenerativeAI(google_api_key=ai_api_key, 
                                model=online_search_agent_config['llm_config']['model_name'],
                                temperature=online_search_agent_config['llm_config']['temperature'])
    
    code_execution_tool = types.Tool(code_execution=types.ToolCodeExecution)

    tools_by_name = {'code_execution_tool': code_execution_tool}
    llm_with_tools = llm.bind_tools(tools_by_name.values())


    system_prompt = online_search_agent_config['prompt_config']['system_message']

    return AgentInitData(tools_by_name=tools_by_name, 
                         llm_with_tools=llm_with_tools,
                         system_prompt=system_prompt)

@entrypoint()
def code_agent(messages: list[BaseMessage]):
    code_agent = _init_code_agent()

    return _tool_calling_agent(code_agent, messages)