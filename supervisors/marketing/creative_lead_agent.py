from langgraph.graph import StateGraph, START
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from agents.marketing.creative_team.copywriter_agent import copywriter_tools
from agents.marketing.creative_team.designer_agent import designer_tools
from agents.marketing.creative_team.creative_lead_agent import creative_lead_tools
from utils.graph_helpers import make_supervisor_node, State

llm = ChatOpenAI(model="gpt-4o")

creative_lead_agent = create_react_agent(llm, tools=creative_lead_tools)

def creative_lead_node(state: State) -> Command:
    result = creative_lead_agent.invoke(state)
    return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="creative_lead")]}, goto="supervisor")

copywriter_agent = create_react_agent(llm, tools=copywriter_tools)

def copywriter_node(state: State) -> Command:
    result = copywriter_agent.invoke(state)
    return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="copywriter")]}, goto="supervisor")

designer_agent = create_react_agent(llm, tools=designer_tools)

def designer_node(state: State) -> Command:
    result = designer_agent.invoke(state)
    return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="designer")]}, goto="supervisor")

creative_supervisor = make_supervisor_node(llm, ["creative_lead", "copywriter", "designer"])

creative_team = StateGraph(State)
creative_team.add_node("supervisor", creative_supervisor)
creative_team.add_node("creative_lead", creative_lead_node)
creative_team.add_node("copywriter", copywriter_node)
creative_team.add_node("designer", designer_node)
creative_team.add_edge(START, "supervisor")
creative_graph = creative_team.compile()
