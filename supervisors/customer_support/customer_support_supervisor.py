from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

from agents.customer_support.faq_agent import FAQAgent
from agents.customer_support.billing_agent import BillingAgent
from agents.customer_support.tech_support_agent import TechSupportAgent

model = ChatOpenAI(model="gpt-4o-mini")

# Wrappers for your agent logic
def faq_tool(query: str) -> str:
    """Handle FAQ-related questions."""
    agent = FAQAgent()
    return agent.choose_tool(query)

def billing_tool(query: str) -> str:
    """Handle billing-related questions."""
    agent = BillingAgent()
    return agent.choose_tool(query)

def tech_support_tool(query: str) -> str:
    """Handle tech support-related questions."""
    agent = TechSupportAgent()
    return agent.choose_tool(query)

# Create specialized agents using create_react_agent
faq_agent = create_react_agent(
    model=model,
    tools=[faq_tool],
    name="faq_agent"
)

billing_agent = create_react_agent(
    model=model,
    tools=[billing_tool],
    name="billing_agent"
)

tech_support_agent = create_react_agent(
    model=model,
    tools=[tech_support_tool],
    name="tech_support_agent"
)

# Create supervisor workflow
supervisor_workflow = create_supervisor(
    [faq_agent, billing_agent, tech_support_agent],
    model=model,
    supervisor_name="customer_support_supervisor",
    prompt=(
        "You are a supervisor managing three agents: faq_agent, billing_agent, and tech_support_agent. "
        "For password, login, or general questions, use faq_agent. "
        "For any question about invoices, payments, billing status, or account charges, use billing_agent. "
        "For technical issues, use tech_support_agent. "
        "If a request involves multiple topics, split it and aggregate the answers from each relevant agent. "
        "Return a single, clear response to the user. Do not include transfer or meta-messages."
    ),
    output_mode="last_message",
)

# Compile the workflow
supervisor_graph = supervisor_workflow.compile()
