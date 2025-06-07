import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@tool
def plan_deployment_strategy(campaign: dict) -> str:
    """Plan deployment strategy for a marketing campaign."""
    # Fake logic: return a simple deployment plan
    name = campaign.get("campaign_name", "Unnamed Campaign") if isinstance(campaign, dict) else "Unnamed Campaign"
    return f"Deployment strategy for {name}: Email, Social Media, and Web."

deployment_lead_tools = [plan_deployment_strategy]

class DeploymentLeadAgent:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the deployment lead. You can ONLY use the following tool: plan_deployment_strategy. If the user's request is not about deployment strategy, you MUST respond with exactly: 'Tool not found'. Do NOT try to be helpful outside your tool. If you are unsure, respond with 'Tool not found'."),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        self.agent = create_tool_calling_agent(self.llm, deployment_lead_tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=deployment_lead_tools, verbose=True)

    def choose_tool(self, question: str) -> str:
        response = self.agent_executor.invoke({"input": question})
        return response["output"]
