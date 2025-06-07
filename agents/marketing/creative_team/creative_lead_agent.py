import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@tool
def review_campaign_assets(assets: list) -> str:
    """Review campaign assets and provide feedback."""
    # Fake logic: return feedback based on asset count
    if not assets:
        return "No assets to review."
    return f"Reviewed {len(assets)} assets. All look good!"

creative_lead_tools = [review_campaign_assets]

class CreativeLeadAgent:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the creative lead. You can ONLY use the following tool: review_campaign_assets. If the user's request is not about reviewing campaign assets, you MUST respond with exactly: 'Tool not found'. Do NOT try to be helpful outside your tool. If you are unsure, respond with 'Tool not found'."),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        self.agent = create_tool_calling_agent(self.llm, creative_lead_tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=creative_lead_tools, verbose=True)

    def choose_tool(self, question: str) -> str:
        response = self.agent_executor.invoke({"input": question})
        return response["output"]
