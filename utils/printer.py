from langchain_core.messages import convert_to_messages

def pretty_print_messages(update):
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return
        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:\n")

    for node_name, node_update in update.items():
        print(f"Update from node {node_name}:\n")
        for m in convert_to_messages(node_update["messages"]):
            m.pretty_print()
        print("\n")
