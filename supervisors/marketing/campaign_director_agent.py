from langgraph.graph import StateGraph, START
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from utils.graph_helpers import make_supervisor_node, State
from analytics_lead_agent import analytics_graph
from creative_lead_agent import creative_graph
from deployment_lead_agent import deployment_graph


llm = ChatOpenAI(model="gpt-4o")
def call_analytics_team(state: State) -> Command:
    response = analytics_graph.invoke({"messages": state["messages"][-1]})
    return Command(update={"messages": [HumanMessage(content=response["messages"][-1].content, name="analytics_team")]}, goto="supervisor")

def call_creative_team(state: State) -> Command:
    response = creative_graph.invoke({"messages": state["messages"][-1]})
    return Command(update={"messages": [HumanMessage(content=response["messages"][-1].content, name="creative_team")]}, goto="supervisor")

def call_deployment_team(state: State) -> Command:
    response = deployment_graph.invoke({"messages": state["messages"][-1]})
    return Command(update={"messages": [HumanMessage(content=response["messages"][-1].content, name="deployment_team")]}, goto="supervisor")

campaign_director_supervisor = make_supervisor_node(llm, ["analytics_team", "creative_team", "deployment_team"])

campaign_director = StateGraph(State)
campaign_director.add_node("supervisor", campaign_director_supervisor)
campaign_director.add_node("analytics_team", call_analytics_team)
campaign_director.add_node("creative_team", call_creative_team)
campaign_director.add_node("deployment_team", call_deployment_team)
campaign_director.add_edge(START, "supervisor")
campaign_graph = campaign_director.compile()
