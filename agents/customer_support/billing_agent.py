from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@tool
def check_invoice_status(invoice_id: str) -> str:
    """Fake tool to check the status of an invoice by its ID."""
    # Fake logic for demonstration
    if invoice_id.startswith("INV"):
        return f"Invoice {invoice_id} is PAID."
    else:
        return f"Invoice {invoice_id} not found."

@tool
def generate_invoice(customer_name: str) -> str:
    """Fake tool to generate an invoice for a customer."""
    # Fake logic for demonstration
    return f"Invoice generated for {customer_name}: INV123456."

billing_tools = [check_invoice_status, generate_invoice]

class BillingAgent:
    def __init__(self):
        """Initializes the BillingAgent with available tools."""
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a billing support agent. "
             "You can ONLY use the following tools: check_invoice_status and generate_invoice. "
             "If the user's request is not about invoices, "
             "you MUST respond with exactly: 'Tool not found'. "
             "Do NOT try to be helpful outside your tools. "
             "If you are unsure, respond with 'Tool not found'."),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        self.agent = create_tool_calling_agent(self.llm, billing_tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=billing_tools, verbose=True)

    def choose_tool(self, question: str) -> str:
        """
        Chooses the appropriate tool based on the input question using LangChain and OpenAI.

        Args:
            question (str): The input question to analyze.

        Returns:
            str: The result from the chosen tool.
        """
        response = self.agent_executor.invoke({"input": question})
        return response["output"]
