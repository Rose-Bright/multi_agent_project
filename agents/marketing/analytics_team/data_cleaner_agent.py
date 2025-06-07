import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@tool
def clean_marketing_data(data: dict) -> dict:
    """Clean and preprocess marketing data."""
    # Fake logic: remove keys with None values and return cleaned data
    if not isinstance(data, dict):
        return {}
    cleaned = {k: v for k, v in data.items() if v is not None}
    cleaned["cleaned"] = True
    return cleaned

data_cleaner_tools = [clean_marketing_data]

class DataCleanerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a data cleaner. You can ONLY use the following tool: clean_marketing_data. If the user's request is not about cleaning marketing data, you MUST respond with exactly: 'Tool not found'. Do NOT try to be helpful outside your tool. If you are unsure, respond with 'Tool not found'."),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        self.agent = create_tool_calling_agent(self.llm, data_cleaner_tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=data_cleaner_tools, verbose=True)

    def choose_tool(self, question: str) -> str:
        response = self.agent_executor.invoke({"input": question})
        return response["output"]
