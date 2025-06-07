# File: campaign_director_supervisor.py

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from utils.graph_helpers import make_supervisor_node, State
from .team_supervisor.analytics_lead_supervisor import analytics_graph
from .team_supervisor.creative_lead_supervisor import creative_graph
from .team_supervisor.deployment_lead_supervisor import deployment_graph

MAX_DEPTH = 3
llm = ChatOpenAI(model="gpt-4o")

def should_terminate(state: State) -> bool:
    return state.get("depth", 0) >= MAX_DEPTH

def call_team(team_name: str, graph, state: State) -> Command:
    prev_depth = state.get("depth", 0)
    depth = prev_depth + 1
    print(f"[CALL_TEAM] {team_name}: prev_depth={prev_depth}, next_depth={depth}")

    if depth > MAX_DEPTH:
        print(f"[CALL_TEAM] {team_name}: MAX_DEPTH {MAX_DEPTH} reached, terminating loop.")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=f"🚫 Recursion limit reached for {team_name}, skipping further delegation.",
                        name=team_name,
                    )
                ] + state["messages"],
                "depth": prev_depth,
                "visited": state.get("visited", []),
            },
            goto="supervisor"
        )

    # Pass full state, but only update messages, depth, and visited in parent
    sub_state = {
        "messages": state["messages"],
        "depth": depth,
        "visited": state.get("visited", []),
    }
    response = graph.invoke(sub_state)

    return Command(
        update={
            "messages": [
                HumanMessage(content=response["messages"][-1].content, name=team_name)
            ] + state["messages"],
            "depth": depth,
            "visited": state.get("visited", []),
        },
        goto="supervisor"
    )


def call_analytics_team(state: State) -> Command:
    return call_team("analytics_team", analytics_graph, state)

def call_creative_team(state: State) -> Command:
    return call_team("creative_team", creative_graph, state)

def call_deployment_team(state: State) -> Command:
    return call_team("deployment_team", deployment_graph, state)

def final_message(state: State) -> Command:
    depth = state.get("depth", 0)
    return Command(
        update={"messages": [
            HumanMessage(content="✅ Campaign complete", name="campaign_director")
        ],
        "depth": depth},
        # No goto; END is implied
    )

campaign_director_supervisor = make_supervisor_node(
    llm, ["analytics_team", "creative_team", "deployment_team"]
)

campaign_director = StateGraph(State)
campaign_director.add_node("supervisor", campaign_director_supervisor)
campaign_director.add_node("analytics_team", call_analytics_team)
campaign_director.add_node("creative_team", call_creative_team)
campaign_director.add_node("deployment_team", call_deployment_team)
campaign_director.add_node("final", final_message)

campaign_director.add_edge(START, "supervisor")
campaign_director.add_edge("final", END)

campaign_graph = campaign_director.compile()
