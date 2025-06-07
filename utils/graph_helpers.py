from typing import Literal
from typing_extensions import TypedDict
from langgraph.types import Command
from langgraph.graph import MessagesState, END
from langchain_core.language_models.chat_models import BaseChatModel

# Inherit from LangGraph MessagesState to hold messages + supervisor decision
class State(MessagesState):
    next: str

# Helper to create a supervisor router node dynamically based on team members
def make_supervisor_node(
    llm: BaseChatModel,
    members: list[str],
    name: str = "supervisor"
):
    """
    Creates a routing supervisor node that decides which agent to call next.

    Args:
        llm: An LLM with structured output (e.g., ChatOpenAI with model="gpt-4o").
        members: List of possible sub-agent names the supervisor can choose from.
        name: Optional name of the supervisor node (for logs, not mandatory).

    Returns:
        A callable function to be used in LangGraph's add_node(...).
    """
    options = ["FINISH"] + members

    system_prompt = (
        f"You are the {name} overseeing these team members: {', '.join(members)}.\n"
        "Based on the user's input and the current conversation, route the task to the most appropriate agent.\n"
        "Return FINISH when all work is complete."
    )

    class Router(TypedDict):
        next: Literal[*options]

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]

        result = llm.with_structured_output(Router).invoke(messages)
        next_agent = result["next"]

        return Command(
            goto=END if next_agent == "FINISH" else next_agent,
            update={"next": next_agent}
        )

    return supervisor_node
