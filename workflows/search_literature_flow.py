from langgraph.func import task, entrypoint
from langchain_core.messages import AIMessage
from langgraph.graph import add_messages

from agents.search_literature_agent.search_literature_agent import search_agent
from agents.search_literature_agent.extract_data_agent import extract_agent
from agents.search_literature_agent.data_cleaner_agent import cleaner_agent
from agents.search_literature_agent.analyze_data_agent import analyze_agent
from agents.search_literature_agent.hypothesis_agent import hypothesis_agent

@task
def call_search_agent(messages):
    return search_agent.invoke({"messages": messages})["messages"]

@task
def call_extract_agent(messages):
    return extract_agent.invoke({"messages": messages})["messages"]

@task
def call_cleaner_agent(messages):
    return cleaner_agent.invoke({"messages": messages})["messages"]

@task
def call_analyze_agent(messages):
    return analyze_agent.invoke({"messages": messages})["messages"]

@task
def call_hypothesis_agent(messages):
    return hypothesis_agent.invoke({"messages": messages})["messages"]

@entrypoint()
def workflow(messages):
    messages = add_messages([], messages)

    call_active_agent = call_search_agent
    while True:
        agent_messages = call_active_agent(messages).result()
        messages = add_messages(messages, agent_messages)
        ai_msg = next(m for m in reversed(agent_messages) if isinstance(m, AIMessage))

        if not ai_msg.tool_calls:
            break

        tool_call = ai_msg.tool_calls[-1]
        name = tool_call["name"]
        call_active_agent = {
            "transfer_to_extract_agent": call_extract_agent,
            "transfer_to_cleaner_agent": call_cleaner_agent,
            "transfer_to_analyze_agent": call_analyze_agent,
            "transfer_to_hypothesis_agent": call_hypothesis_agent,
        }.get(name)

        if not call_active_agent:
            break

    return messages
