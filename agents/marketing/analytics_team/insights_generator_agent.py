import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@tool
def generate_insights(data: dict) -> str:
    """Generate marketing insights from cleaned data."""
    # Fake logic: generate a simple insight based on data
    if not data or not isinstance(data, dict):
        return "No data to generate insights."
    if data.get("cleaned"):
        return "Insight: Data is clean and ready for analysis."
    return "Insight: Data may need cleaning."

insights_tools = [generate_insights]

class InsightsGeneratorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an insights generator. You can ONLY use the following tool: generate_insights. If the user's request is not about generating insights, you MUST respond with exactly: 'Tool not found'. Do NOT try to be helpful outside your tool. If you are unsure, respond with 'Tool not found'."),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        self.agent = create_tool_calling_agent(self.llm, insights_tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=insights_tools, verbose=True)

    def choose_tool(self, question: str) -> str:
        response = self.agent_executor.invoke({"input": question})
        return response["output"]
