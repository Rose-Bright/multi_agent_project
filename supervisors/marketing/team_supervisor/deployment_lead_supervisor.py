from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from agents.marketing.deployment_team.email_agent import email_tools
from agents.marketing.deployment_team.social_agent import social_tools
from agents.marketing.deployment_team.deployment_lead_agent import deployment_lead_tools
from utils.graph_helpers import make_supervisor_node, State, build_agent_node, make_finish_node

llm = ChatOpenAI(model="gpt-4o")

deployment_lead_agent = create_react_agent(llm, tools=deployment_lead_tools)
email_agent = create_react_agent(llm, tools=email_tools)
social_agent = create_react_agent(llm, tools=social_tools)

deployment_lead_node = build_agent_node(deployment_lead_agent, "deployment_lead")
email_node = build_agent_node(email_agent, "email")
social_node = build_agent_node(social_agent, "social")

finish_node = make_finish_node("✅ Deployment complete", "system")

deployment_supervisor = make_supervisor_node(llm, [
    "deployment_lead", "email", "social"
])

deployment_team = StateGraph(State)
deployment_team.add_node("supervisor", deployment_supervisor)
deployment_team.add_node("deployment_lead", deployment_lead_node)
deployment_team.add_node("email", email_node)
deployment_team.add_node("social", social_node)
deployment_team.add_node("final", finish_node)

deployment_team.add_edge(START, "supervisor")

# Route each agent node back to supervisor
deployment_team.add_edge("deployment_lead", "supervisor")
deployment_team.add_edge("email", "supervisor")
deployment_team.add_edge("social", "supervisor")

# Allow supervisor to send to finish
deployment_team.add_edge("supervisor", "final")

deployment_team.add_edge("final", END)

deployment_graph = deployment_team.compile()
