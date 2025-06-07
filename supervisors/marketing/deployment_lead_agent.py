from langgraph.graph import StateGraph, START
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from agents.marketing.deployment_team.email_agent import email_tools
from agents.marketing.deployment_team.social_agent import social_tools
from agents.marketing.deployment_team.deployment_lead_agent import deployment_lead_tools
from utils.graph_helpers import make_supervisor_node, State

llm = ChatOpenAI(model="gpt-4o")

deployment_lead_agent = create_react_agent(llm, tools=deployment_lead_tools)

def deployment_lead_node(state: State) -> Command:
    result = deployment_lead_agent.invoke(state)
    return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="deployment_lead")]}, goto="supervisor")

email_agent = create_react_agent(llm, tools=email_tools)

def email_node(state: State) -> Command:
    result = email_agent.invoke(state)
    return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="email")]}, goto="supervisor")

social_agent = create_react_agent(llm, tools=social_tools)

def social_node(state: State) -> Command:
    result = social_agent.invoke(state)
    return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="social")]}, goto="supervisor")

deployment_supervisor = make_supervisor_node(llm, ["deployment_lead", "email", "social"])

deployment_team = StateGraph(State)
deployment_team.add_node("supervisor", deployment_supervisor)
deployment_team.add_node("deployment_lead", deployment_lead_node)
deployment_team.add_node("email", email_node)
deployment_team.add_node("social", social_node)
deployment_team.add_edge(START, "supervisor")
deployment_graph = deployment_team.compile()
