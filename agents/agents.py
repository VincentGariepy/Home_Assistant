from contextlib import asynccontextmanager
import os
import types
import yaml
from datetime import date
from retry import retry
from typing import Literal

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
from langchain_experimental.utilities import PythonREPL
from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchResults
from google import genai

from tools.utils import ADDRESSES
from tools.places_tools import get_route
from tools.weather_tools import get_current_weather, get_daily_weather, get_hourly_weather
from tools.timer_tool import add_timer, timer_data, convert_hours_to_seconds, convert_minutes_to_seconds
from agents.agent_init_data import AgentInitData
from agents.utils import extract_last_ai_message
from log_config.logging_config import get_conversation_logger, logger

AI_API_KEY = os.getenv('GOOGLE_AI_KEY')
MAPS_API_KEY = os.getenv('GOOGLE_MAPS_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

with open('agents/agents_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

_conversation_logger = get_conversation_logger()


@retry(tries=3, backoff=2, jitter=(2, 5))
@task
def _call_llm(messages: list[BaseMessage], agent_data: AgentInitData, agent_name: str = "unknown"):
    """LLM decides whether to call a tool or not"""
    if type(messages) == str:
        messages = [HumanMessage(content=messages)]

    response = agent_data.llm_with_tools.invoke(
        [SystemMessage(content=agent_data.system_prompt)] + messages
    )
    logger.debug(f"LLM response for {agent_name}: {response}")
    # Extract and log token usage if available
    try:
        if hasattr(response, 'response_metadata') and hasattr(response, 'usage_metadata'):
            metadata = response.response_metadata
            usage = response.usage_metadata
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            reasoning_tokens = usage.get('output_token_details', {}).get('reasoning', 0)
            total_tokens = input_tokens + output_tokens
            model_name = metadata.get('model_name', 'unknown')
            
            _conversation_logger.log_token_usage(
                agent_name=agent_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                total_tokens=total_tokens,
                model_name=model_name,
            )
            logger.debug(f"Token usage for {agent_name}: input={input_tokens}, output={output_tokens}, reasoning={reasoning_tokens}, total={total_tokens}")
    except Exception as e:
        logger.debug(f"Could not extract token usage: {e}")
    
    return response

@task
def _call_tool(tool_call: ToolCall, agent_data: AgentInitData):
    """Performs the tool call"""
    tool = agent_data.tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)

def _tool_calling_agent(agent_data: AgentInitData, messages: list[BaseMessage] | str, agent_name: str = "unknown"):
    if type(messages) == str:
        messages = [HumanMessage(content=messages)]
    request = messages[-1].content if messages else ""
    llm_response = _call_llm(messages, agent_data, agent_name).result()

    tool_results = []
    tool_calls = []
    while True:
        if not llm_response.tool_calls:  
            break
        
        tool_calls += llm_response.tool_calls
        # Execute tools
        tool_result_futures = [
            _call_tool(tool_call, agent_data) for tool_call in llm_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [llm_response, *tool_results])
        llm_response = _call_llm(messages, agent_data, agent_name).result()
    
    if tool_results:
        if agent_name.startswith("main_agent"):
            parsed_tool_results = [extract_last_ai_message(tool_results[-1].content)]
        else:
            parsed_tool_results = [tool_result.content for tool_result in tool_results]
    else:
        parsed_tool_results = []


    _conversation_logger.log_agent_response(
        agent_name=agent_name,
        request=request,
        response=llm_response.text,
        tool_calls=tool_calls,
        tool_call_results=parsed_tool_results,
    )    
    messages = add_messages(messages, llm_response)
    return messages

def _init_route_agent():
    route_agent_config = config['route_agent']
    llm = ChatGoogleGenerativeAI(google_api_key=AI_API_KEY, 
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
def route_agent(messages: list[BaseMessage] | str):
    route_agent_data = _init_route_agent()
    logger.info("Route agent called")
    return _tool_calling_agent(route_agent_data, messages, agent_name="route_agent")
   

def _init_weather_agent():
    weather_agent_config = config['weather_agent']
    llm = ChatGoogleGenerativeAI(google_api_key=AI_API_KEY, 
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
def weather_agent(messages: list[BaseMessage] | str):
    weather_agent_data = _init_weather_agent()
    logger.info("Weather agent called")
    return _tool_calling_agent(weather_agent_data, messages, agent_name="weather_agent")

def _init_online_search_agent():
    online_search_agent_config = config['online_search_agent']
    llm = ChatGoogleGenerativeAI(google_api_key=AI_API_KEY, 
                                model=online_search_agent_config['llm_config']['model_name'],
                                temperature=online_search_agent_config['llm_config']['temperature'])

    search = GoogleSearchAPIWrapper(google_api_key=MAPS_API_KEY, 
                                    GOOGLE_CSE_ID=GOOGLE_CSE_ID)

    grounding_tool = GoogleSearchResults(api_wrapper=search)
    
    tools = [grounding_tool]

    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)


    system_prompt = online_search_agent_config['prompt_config']['system_message'].format(
                                    date.today().strftime('%Y-%m-%d'))

    return AgentInitData(tools_by_name=tools_by_name, 
                         llm_with_tools=llm_with_tools,
                         system_prompt=system_prompt)

@entrypoint()
def online_search_agent(messages: list[BaseMessage] | str):
    online_search_agent_data = _init_online_search_agent()
    logger.info("Online search agent called")
    return _tool_calling_agent(online_search_agent_data, messages, agent_name="online_search_agent")

def _init_code_agent():
    code_agent_config = config['code_agent']
    llm = ChatGoogleGenerativeAI(google_api_key=AI_API_KEY, 
                                model=code_agent_config['llm_config']['model_name'],
                                temperature=code_agent_config['llm_config']['temperature'])
    
    python_repl = PythonREPL()
    # You can create the tool to pass to an agent
    repl_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    )
    tools = [repl_tool]

    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)


    system_prompt = code_agent_config['prompt_config']['system_message']

    return AgentInitData(tools_by_name=tools_by_name, 
                         llm_with_tools=llm_with_tools,
                         system_prompt=system_prompt)

@entrypoint()
def code_agent(messages: list[BaseMessage] | str):
    code_agent_data = _init_code_agent()
    logger.info("Code agent called")
    return _tool_calling_agent(code_agent_data, messages, agent_name="code_agent")

def _init_timer_agent():
    timer_agent_config = config['timer_agent']
    llm = ChatGoogleGenerativeAI(google_api_key=AI_API_KEY, 
                                model=timer_agent_config['llm_config']['model_name'],
                                temperature=timer_agent_config['llm_config']['temperature'])
    tools = [add_timer, timer_data, convert_hours_to_seconds, convert_minutes_to_seconds]

    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)


    system_prompt = timer_agent_config['prompt_config']['system_message']

    return AgentInitData(tools_by_name=tools_by_name, 
                         llm_with_tools=llm_with_tools,
                         system_prompt=system_prompt)

@entrypoint()
def timer_agent(messages: list[BaseMessage] | str):
    timer_agent_data = _init_timer_agent()
    logger.info("Timer agent called")
    return _tool_calling_agent(timer_agent_data, messages, agent_name="timer_agent")

def _init_quirky_agent(tone: str):
    quirky_agent_config = config['quirky_agent']

    llm = ChatGoogleGenerativeAI(google_api_key=AI_API_KEY, 
                                model=quirky_agent_config['llm_config']['model_name'],
                                temperature=quirky_agent_config['llm_config']['temperature'])
    if tone == 'sassy':
        system_prompt = quirky_agent_config['prompt_config']['system_message_sassy']
    elif tone == 'scottish':
        system_prompt = quirky_agent_config['prompt_config']['system_message_scottish']
    else:
        system_prompt = quirky_agent_config['prompt_config']['system_message_neutral']

    return AgentInitData(tools_by_name={}, 
                         llm_with_tools=llm,
                         system_prompt=system_prompt)

@entrypoint()
def quirky_agent(input: dict):
    quirky_agent_data = _init_quirky_agent(input['tone'])

    return _tool_calling_agent(quirky_agent_data, input['messages'], agent_name=f"quirky_agent_{input['tone']}")

def _create_main_agent_tools():
    route_agent_tool = Tool(
        name="route_agent_tool",
        description="A tool for getting driving route information. Takes a prompt as input.",
        func=route_agent.invoke,
    )

    weather_agent_tool = Tool(
        name="weather_agent_tool",
        description="A tool for getting weather information. Takes a prompt as input.",
        func=weather_agent.invoke,
    )

    online_search_agent_tool = Tool(
        name="online_search_agent_tool",
        description="A tool for getting online search information. Takes a prompt as input.",
        func=online_search_agent.invoke,
    )

    code_agent_tool = Tool(
        name="code_agent_tool",
        description="A tool for getting code execution information. Takes a prompt as input.",
        func=code_agent.invoke,
    )
    timer_agent_tool = Tool(
        name="timer_agent_tool",
        description="A tool for managing timers. Takes a prompt as input.",
        func=timer_agent.invoke,
    )

    return [route_agent_tool, weather_agent_tool, online_search_agent_tool, code_agent_tool, timer_agent_tool]

def _init_main_agent(tone: str):
    main_agent_config = config['main_agent']
    llm = ChatGoogleGenerativeAI(google_api_key=AI_API_KEY, 
                                model=main_agent_config['llm_config']['model_name'],
                                temperature=main_agent_config['llm_config']['temperature'])
    
    tools = _create_main_agent_tools()

    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    if tone == 'sassy':
        system_prompt = main_agent_config['prompt_config']['system_message_sassy']
    elif tone == 'scottish':
        system_prompt = main_agent_config['prompt_config']['system_message_scottish']
    elif tone == 'jarjar':
        system_prompt = main_agent_config['prompt_config']['system_message_jarjar']
    else:
        system_prompt = main_agent_config['prompt_config']['system_message_neutral']

    system_prompt = system_prompt.format(list(tools_by_name.keys()),
                                    date.today().strftime('%Y-%m-%d'))

    return AgentInitData(tools_by_name=tools_by_name, 
                         llm_with_tools=llm_with_tools,
                         system_prompt=system_prompt)

@entrypoint()
def main_agent(input: dict):
    tone = input.get("tone", "neutral")
    main_agent_inst = _init_main_agent(tone=tone)
    logger.info(f"Main agent called with tone: {tone}")
    return _tool_calling_agent(main_agent_inst, input["messages"], agent_name=f"main_agent_{tone}")

