from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from agents.marketing.analytics_team.analytics_lead_agent import analytics_lead_tools
from agents.marketing.analytics_team.data_cleaner_agent import data_cleaner_tools
from agents.marketing.analytics_team.insights_generator_agent import insights_tools
from utils.graph_helpers import make_supervisor_node, State, build_agent_node, make_finish_node

llm = ChatOpenAI(model="gpt-4o")

analytics_lead_agent = create_react_agent(llm, tools=analytics_lead_tools)
data_cleaner_agent = create_react_agent(llm, tools=data_cleaner_tools)
insights_generator_agent = create_react_agent(llm, tools=insights_tools)

analytics_lead_node = build_agent_node(analytics_lead_agent, "analytics_lead")
data_cleaner_node = build_agent_node(data_cleaner_agent, "data_cleaner")
insights_generator_node = build_agent_node(insights_generator_agent, "insights_generator")

finish_node = make_finish_node("✅ Analytics process complete.", "system")

analytics_supervisor = make_supervisor_node(llm, [
    "analytics_lead", "data_cleaner", "insights_generator"
])

analytics_team = StateGraph(State)
analytics_team.add_node("supervisor", analytics_supervisor)
analytics_team.add_node("analytics_lead", analytics_lead_node)
analytics_team.add_node("data_cleaner", data_cleaner_node)
analytics_team.add_node("insights_generator", insights_generator_node)
analytics_team.add_node("final", finish_node)

# Graph edges
analytics_team.add_edge(START, "supervisor")

# Make sure all agents route back to the supervisor unless depth limit is hit
analytics_team.add_edge("analytics_lead", "supervisor")
analytics_team.add_edge("data_cleaner", "supervisor")
analytics_team.add_edge("insights_generator", "supervisor")

# Allow supervisor to send to finish
analytics_team.add_edge("supervisor", "final")

# Mark end
analytics_team.add_edge("final", END)

analytics_graph = analytics_team.compile()
