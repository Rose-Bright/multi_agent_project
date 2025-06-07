import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@tool
def design_visual_asset(description: str) -> str:
    """Design a visual asset based on a description."""
    # Fake logic: return a confirmation message
    if not description:
        return "No description provided."
    return f"Visual asset created for: {description}"

designer_tools = [design_visual_asset]

class DesignerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a designer. You can ONLY use the following tool: design_visual_asset. If the user's request is not about designing a visual asset, you MUST respond with exactly: 'Tool not found'. Do NOT try to be helpful outside your tool. If you are unsure, respond with 'Tool not found'."),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        self.agent = create_tool_calling_agent(self.llm, designer_tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=designer_tools, verbose=True)

    def choose_tool(self, question: str) -> str:
        response = self.agent_executor.invoke({"input": question})
        return response["output"]
