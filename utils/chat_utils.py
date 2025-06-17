from langchain_core.messages import HumanMessage, AIMessage
import logging
logger = logging.getLogger(__name__)

def convert_messages(messages):
    """Convert dict messages to LangChain message objects."""
    langchain_messages = []
    for msg in messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
    return langchain_messages

def retrieve_relevant_memories(memory_manager, user_id, query):
    """Retrieve relevant long-term memories as a string."""
    relevant_memories = memory_manager.search_long_term_memory(user_id, query)
    logger.info(f"Retrieved {relevant_memories} relevant memories for query: {query}")
    return "\n".join([m["text"] for m in relevant_memories])

def inject_memories(langchain_messages, memories_text):
    """Inject memories into the message list if any exist."""
    if memories_text:
        langchain_messages.insert(0, HumanMessage(content=f"Relevant past knowledge:\n{memories_text}"))
    return langchain_messages

def handle_agent_chat(agent, messages, thread_id, memory_manager, user_id="123"):
    chat_history = memory_manager.get_latest_chat_history(thread_id)
    new_messages = convert_messages(messages)
    result = agent.agent_executor.invoke(
        {"input": messages[-1]["content"], "chat_history": chat_history},
        config={"configurable": {"thread_id": thread_id, "checkpoint_ns": f"{thread_id}-ns"}}
    )
    new_messages.append(AIMessage(content=result["output"]))
    memory_manager.save_chat_history(thread_id, new_messages)
    # Save with role and agent name
    memory_manager.write_long_term_memory(
        user_id,
        f"Q: {messages[-1]['content']} A: {result['output']}",
        role="assistant",
        agent_name=type(agent).__name__
    )
    return result["output"]

def prepare_agent_input(agent, messages, memory_manager, user_id, config=None):
    if config is None:
        config = {}

    langchain_messages = convert_messages(messages)
    latest_input = messages[-1]["content"]
    memories_text = retrieve_relevant_memories(memory_manager, user_id, latest_input)
    langchain_messages = inject_memories(langchain_messages, memories_text)
    config["configurable"]["chat_history"] = langchain_messages
    print(f"config: {config}")
    response = agent.agent_executor.invoke({"input": latest_input}, config=config)
    return response["output"]
