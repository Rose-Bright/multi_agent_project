import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@tool
def send_marketing_email(content: str, recipients: list) -> str:
    """Send a marketing email to a list of recipients."""
    # Fake logic: return a message with recipient count
    if not recipients:
        return "No recipients provided."
    return f"Email sent to {len(recipients)} recipients."

email_tools = [send_marketing_email]

class EmailAgent:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an email agent. You can ONLY use the following tool: send_marketing_email. If the user's request is not about sending marketing emails, you MUST respond with exactly: 'Tool not found'. Do NOT try to be helpful outside your tool. If you are unsure, respond with 'Tool not found'."),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        self.agent = create_tool_calling_agent(self.llm, email_tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=email_tools, verbose=True)

    def choose_tool(self, question: str) -> str:
        response = self.agent_executor.invoke({"input": question})
        return response["output"]
