from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from agents.marketing.creative_team.copywriter_agent import copywriter_tools
from agents.marketing.creative_team.designer_agent import designer_tools
from agents.marketing.creative_team.creative_lead_agent import creative_lead_tools
from utils.graph_helpers import make_supervisor_node, State, build_agent_node, make_finish_node

llm = ChatOpenAI(model="gpt-4o")

creative_lead_agent = create_react_agent(llm, tools=creative_lead_tools)
copywriter_agent = create_react_agent(llm, tools=copywriter_tools)
designer_agent = create_react_agent(llm, tools=designer_tools)

creative_lead_node = build_agent_node(creative_lead_agent, "creative_lead")
copywriter_node = build_agent_node(copywriter_agent, "copywriter")
designer_node = build_agent_node(designer_agent, "designer")

finish_node = make_finish_node("✅ Creative process complete.", "system")

creative_supervisor = make_supervisor_node(llm, [
    "creative_lead", "copywriter", "designer"
])

creative_team = StateGraph(State)
creative_team.add_node("supervisor", creative_supervisor)
creative_team.add_node("creative_lead", creative_lead_node)
creative_team.add_node("copywriter", copywriter_node)
creative_team.add_node("designer", designer_node)
creative_team.add_node("final", finish_node)

creative_team.add_edge(START, "supervisor")

# Route each agent node back to supervisor
creative_team.add_edge("creative_lead", "supervisor")
creative_team.add_edge("copywriter", "supervisor")
creative_team.add_edge("designer", "supervisor")

# Allow supervisor to send to finish
creative_team.add_edge("supervisor", "final")

creative_team.add_edge("final", END)

creative_graph = creative_team.compile()
