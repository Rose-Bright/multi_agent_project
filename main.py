from supervisors.customer_support.customer_support_supervisor import supervisor_graph
from supervisors.marketing.campaign_director_supervisor import campaign_graph  # Add this import

from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage

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
    #query = {
    #"role": "user",
    #"content": "I'd like to search recent research on CRISPR in cancer. Clean and analyze the data, then generate a hypothesis."
    #}

    #for chunk in workflow.stream([query], subgraphs=True):
    #    pretty_print_messages(chunk)

    print("\n--- Campaign Director Supervisor Test ---")
    campaign_test_messages = [
        "Launch a new marketing campaign for our summer product line.",
        "Analyze the last campaign's performance.",
        "Deploy the new ad creatives to all channels."
    ]

    for msg in campaign_test_messages:
        print(f"\nUser: {msg}")
        state = MessagesState(messages=[HumanMessage(content=msg)])
        result = campaign_graph.invoke(state)
        print("\n--- Full Message Trace ---")
        for m in result["messages"]:
            m.pretty_print()

        # Find the last AI message
        print("\n--- AI Response ---")
        ai_messages = [m for m in result["messages"] if m.type == "ai"]
        if ai_messages:
            print(f"AI: {ai_messages[-1].content}")
        else:
            print("No AI response.")
