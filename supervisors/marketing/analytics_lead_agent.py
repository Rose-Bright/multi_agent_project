from langgraph.graph import StateGraph, START
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from agents.marketing.analytics_team.analytics_lead_agent import analytics_lead_tools
from agents.marketing.analytics_team.data_cleaner_agent import data_cleaner_tools
from agents.marketing.analytics_team.insights_generator_agent import insights_tools
from utils.graph_helpers import make_supervisor_node, State

llm = ChatOpenAI(model="gpt-4o")

analytics_lead_agent = create_react_agent(llm, tools=analytics_lead_tools)

def analytics_lead_node(state: State) -> Command:
    result = analytics_lead_agent.invoke(state)
    return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="analytics_lead")]}, goto="supervisor")

data_cleaner_agent = create_react_agent(llm, tools=data_cleaner_tools)

def data_cleaner_node(state: State) -> Command:
    result = data_cleaner_agent.invoke(state)
    return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="data_cleaner")]}, goto="supervisor")

insights_generator_agent = create_react_agent(llm, tools=insights_tools)

def insights_generator_node(state: State) -> Command:
    result = insights_generator_agent.invoke(state)
    return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="insights_generator")]}, goto="supervisor")

analytics_supervisor = make_supervisor_node(llm, ["analytics_lead", "data_cleaner", "insights_generator"])

analytics_team = StateGraph(State)
analytics_team.add_node("supervisor", analytics_supervisor)
analytics_team.add_node("analytics_lead", analytics_lead_node)
analytics_team.add_node("data_cleaner", data_cleaner_node)
analytics_team.add_node("insights_generator", insights_generator_node)
analytics_team.add_edge(START, "supervisor")
analytics_graph = analytics_team.compile()
