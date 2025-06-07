import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@tool
def analyze_campaign_performance(data: dict) -> str:
    """Analyze campaign performance data and return insights."""
    # Fake logic: pretend to analyze campaign data and return a summary
    if not data:
        return "No campaign data provided."
    campaign_name = data.get("campaign_name", "Unknown Campaign")
    impressions = data.get("impressions", 0)
    clicks = data.get("clicks", 0)
    ctr = (clicks / impressions * 100) if impressions else 0
    return f"Analysis for {campaign_name}: Impressions={impressions}, Clicks={clicks}, CTR={ctr:.2f}%."

analytics_lead_tools = [analyze_campaign_performance]

class AnalyticsLeadAgent:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the analytics lead. You can ONLY use the following tool: analyze_campaign_performance. If the user's request is not about campaign analysis, you MUST respond with exactly: 'Tool not found'. Do NOT try to be helpful outside your tool. If you are unsure, respond with 'Tool not found'."),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        self.agent = create_tool_calling_agent(self.llm, analytics_lead_tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=analytics_lead_tools, verbose=True)

    def choose_tool(self, question: str) -> str:
        response = self.agent_executor.invoke({"input": question})
        return response["output"]
