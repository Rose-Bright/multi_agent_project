from typing_extensions import TypedDict
from langgraph.types import Command
from langgraph.graph import MessagesState, END
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

# Inherit from LangGraph MessagesState to hold messages + supervisor decision
class State(MessagesState):
    next: str
    depth: int
    visited: list[str]  # Track visited agents

# Helper to create a supervisor router node dynamically based on team members
def make_supervisor_node(
    llm: BaseChatModel,
    members: list[str],
    name: str = "supervisor",
    max_depth: int = 10
):
    def supervisor_node(state: State) -> Command:
        depth = state.get("depth", 0) + 1
        visited = state.get("visited", [])
        if depth > max_depth:
            print(f"[Supervisor {name}] Depth: {depth} (MAX REACHED), Ending.")
            return Command(goto=END, update={"depth": depth, "visited": visited})

        # Filter out already visited options
        available_options = [m for m in members if m not in visited]
        if not available_options:
            return Command(goto="final", update={"depth": depth, "visited": visited})

        #options = ["FINISH"] + available_options

        system_prompt = (
            f"You are the {name} overseeing these team members: {', '.join(members)}.\n"
            f"Agents already visited: {', '.join(visited)}.\n"
            "Pick the next appropriate agent. Return FINISH when all work is complete."
        )

        # Use a generic TypedDict for router output
        class Router(TypedDict):
            next: str

        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]

        result = llm.with_structured_output(Router).invoke(messages)
        next_agent = result["next"]
        print(f"[Supervisor {name}] Depth: {depth}, Next agent: {next_agent}")
        new_visited = visited + [next_agent] if next_agent != "FINISH" else visited
        return Command(
            goto=END if next_agent == "FINISH" else next_agent,
            update={"next": next_agent, "depth": depth, "visited": new_visited}
        )

    return supervisor_node

# Patch: agent nodes and finish node do NOT update depth

def build_agent_node(agent, name, max_depth=3):
    def node(state: State) -> Command:
        # depth = state.get("depth", 0) + 1  # REMOVE depth increment
        print(f"[Agent {name}] (no depth increment)")
        # if depth > max_depth:
        #     return Command(goto="final", update={"depth": depth})
        result = agent.invoke(state)
        updated_messages = state["messages"] + [
            HumanMessage(content=result["messages"][-1].content, name=name)
        ]
        next_node = "supervisor"  # always route back to supervisor
        return Command(update={"messages": updated_messages}, goto=next_node)
    return node

def make_finish_node(message: str, name: str = "system"):
    def finish_node(state: State) -> Command:
        # depth = state.get("depth", 0)  # REMOVE depth
        return Command(
            update={
                "messages": state["messages"] + [
                    HumanMessage(content=message, name=name)
                ]
            },
            goto=END
        )
    return finish_node
