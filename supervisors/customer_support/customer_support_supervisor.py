from vertexai import agent_engines

from agents.customer_support.faq_agent import FAQAgent
from agents.customer_support.billing_agent import BillingAgent
from agents.customer_support.tech_support_agent import TechSupportAgent

model = "gemini-2.0-flash"

def faq_tool(query: str) -> str:
    """Handles FAQ-related questions."""
    agent = FAQAgent()
    return agent.choose_tool(query)

def billing_tool(query: str) -> str:
    """Handles billing-related questions."""
    agent = BillingAgent()
    return agent.choose_tool(query)

def tech_support_tool(query: str) -> str:
    """Handles technical support-related questions."""
    agent = TechSupportAgent()
    return agent.choose_tool(query)

system_instruction = (
    "You are a supervisor managing three tools: faq_tool, billing_tool, and tech_support_tool. "
    "For password, login, or general questions, use faq_tool. "
    "For any question about invoices, payments, billing status, or account charges, use billing_tool. "
    "For technical issues, use tech_support_tool. "
    "If a request involves multiple topics, split it and aggregate the answers from each relevant tool. "
    "Return a single, clear response to the user. Do not include transfer or meta-messages."
)

#custom_prompt_template = {
#    "user_input": lambda x: x["input"],
##    "history": lambda x: x["history"],
 #   "agent_scratchpad": lambda x: format_to_tool_messages(x["intermediate_steps"]),
#} | ChatPromptTemplate.from_messages([
 #   ("system", system_instruction),
 #   ("placeholder", "{history}"),
 #   ("user", "{user_input}"),
 #   ("placeholder", "{agent_scratchpad}"),
#])

agent = agent_engines.LanggraphAgent(
    model=model,
    tools=[faq_tool, billing_tool, tech_support_tool],
    #system_instruction=system_instruction,
)
