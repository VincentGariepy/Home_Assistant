def live_main_agent_direct_tools():
    search = GoogleSearchAPIWrapper(google_api_key=MAPS_API_KEY, 
                                    GOOGLE_CSE_ID=GOOGLE_CSE_ID)
    grounding = GoogleSearchResults(api_wrapper=search)
    grounding_tool = Tool(
        name="grounding_tool",
        description="A tool for getting online search information. Takes a prompt as input.",
        func=grounding.invoke,
    )
    python_repl = PythonREPL()
    # You can create the tool to pass to an agent
    repl_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    )

    return [
            genai.types.Tool(
                name="get_route",
                description="A tool for getting driving route information. Takes a destination adress string and origin adress string as input.",
                func=get_route
            ),
            genai.types.Tool(
                name="get_current_weather",
                description="A tool for getting current weather information. Takes an address string as input.",
                func=get_current_weather
            ),
            genai.types.Tool(
                name="get_daily_weather",
                description="A tool for getting daily weather information. Takes an address string as input.",
                func=get_daily_weather
            ),
            genai.types.Tool(
                name="get_hourly_weather",
                description="A tool for getting hourly weather information. Takes an address string as input.",
                func=get_hourly_weather
            ),
            genai.types.Tool(
                name="grounding_tool",
                description="A tool for getting online search information. Takes a Google search prompt as input.",
                func=grounding_tool,
            ),
            genai.types.Tool(
                name="convert_hours_to_seconds",
                description="A tool for converting hours to seconds. Takes a number of hours as input.",
                func=convert_hours_to_seconds,
            ),
            genai.types.Tool(
                name="convert_minutes_to_seconds",
                description="A tool for converting minutes to seconds. Takes a number of minutes as input.",
                func=convert_minutes_to_seconds,
            ),
            genai.types.Tool(
                name="add_timer",
                description="A tool for adding a timer. Takes a duration in seconds and a label as input.",
                func=add_timer,
            ),
            genai.types.Tool(
                name="timer_data",
                description="A tool for getting current data for all timers. Takes a no input.",
                func=timer_data,
            ),
            genai.types.Tool(
                name="code_tool",
                description="A tool for getting code execution information. Takes python code as input.",
                func=repl_tool,
            )
        ]

async def live_main_agent_direct_tools(tone: Literal["neutral", "sassy", "scottish", "jarjar"] = "neutral", response_mode: Literal["TEXT", "AUDIO"] = "TEXT"):
    client = genai.Client(api_key=AI_API_KEY)
    model = config['live_agent_direct']['llm_config']['model_name']
    
    tools = live_main_agent_direct_tools()

    config = genai.types.LiveConfig(
        tools=tools,
        system_prompt=config['live_agent_direct']['prompt_config'][f'system_message_{tone}'].format(
                                    list({"tool_name" : tool.name, "tool_description" : tool.description} for tool in tools),
                                    date.today().strftime('%Y-%m-%d'),
                                    str(list(ADDRESSES.keys()))),
        response_mode=response_mode,
        temperature=config['live_agent_direct']['llm_config']['temperature'],
    )

    async with client.aio.live.connect(model=model, config=config) as session:
        yield session
