from supervisors.customer_support.customer_support_supervisor import supervisor_graph
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
from workflows.search_literature_flow import workflow
from utils.printer import pretty_print_messages

if __name__ == "__main__":
    print("\n--- Customer Support Supervisor Test ---")
    test_messages = [
       # "What is your return policy?",
       # "My computer won't turn on.",
       # "I have a question about my invoice.",
       # "How do I reset my password? Could I have the status of my invoice that start with INV123456?",
       # "Can you translate this to Spanish?"
    ]

    for msg in test_messages:
        print(f"\nUser: {msg}")
        state = MessagesState(messages=[HumanMessage(content=msg)])
        result = supervisor_graph.invoke(state)
        print("\n--- Full Message Trace ---")
        for m in result["messages"]:
            m.pretty_print()

        # Find the last AI message
        print("\n--- AI Response ---")
        for m in result["messages"]:
            ai_messages = [m for m in result["messages"] if m.type == "ai"]
        if ai_messages:
            print(f"AI: {ai_messages[-1].content}")
        else:
            print("No AI response.")

    print("\n--- Research Literature Workflow Test ---")
    query = {
    "role": "user",
    "content": "I'd like to search recent research on CRISPR in cancer. Clean and analyze the data, then generate a hypothesis."
    }

    for chunk in workflow.stream([query], subgraphs=True):
        pretty_print_messages(chunk)
