import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@tool
def write_ad_copy(prompt: str) -> str:
    """Write creative ad copy based on a prompt."""
    # Fake logic: return a templated ad copy
    if not prompt:
        return "No prompt provided."
    return f"Don't miss out! {prompt} Get yours today!"

copywriter_tools = [write_ad_copy]

class CopywriterAgent:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a copywriter. You can ONLY use the following tool: write_ad_copy. If the user's request is not about writing ad copy, you MUST respond with exactly: 'Tool not found'. Do NOT try to be helpful outside your tool. If you are unsure, respond with 'Tool not found'."),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        self.agent = create_tool_calling_agent(self.llm, copywriter_tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=copywriter_tools, verbose=True)

    def choose_tool(self, question: str) -> str:
        response = self.agent_executor.invoke({"input": question})
        return response["output"]
