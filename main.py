from supervisors.customer_support.customer_support_supervisor import supervisor_graph
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print("\n--- Customer Support Supervisor Test ---")
    test_messages = [
       # "What is your return policy?",
       # "My computer won't turn on.",
       # "I have a question about my invoice.",
        "How do I reset my password? Could I have the status of my invoice that start with INV123456?",
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
