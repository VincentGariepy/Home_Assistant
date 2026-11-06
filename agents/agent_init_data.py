class AgentInitData:
    def __init__(self, tools_by_name, llm_with_tools, system_prompt):
        self.tools_by_name = tools_by_name
        self.llm_with_tools = llm_with_tools
        self.system_prompt = system_prompt