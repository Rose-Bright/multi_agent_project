import unittest
from supervisors.customer_support.customer_support_supervisor import supervisor_graph
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage

class TestCustomerSupportSupervisor(unittest.TestCase):
    def test_faq(self):
        msg = "What is your return policy?"
        state = MessagesState(messages=[HumanMessage(content=msg)])
        result = supervisor_graph.invoke(state)
        ai_messages = [m for m in result["messages"] if m.type == "ai"]
        self.assertTrue(any("return" in m.content.lower() for m in ai_messages))

    def test_billing(self):
        msg = "Can you check invoice INV123456?"
        state = MessagesState(messages=[HumanMessage(content=msg)])
        result = supervisor_graph.invoke(state)
        ai_messages = [m for m in result["messages"] if m.type == "ai"]
        self.assertTrue(any("invoice" in m.content.lower() for m in ai_messages))

    def test_tech_support(self):
        msg = "My computer won't turn on."
        state = MessagesState(messages=[HumanMessage(content=msg)])
        result = supervisor_graph.invoke(state)
        ai_messages = [m for m in result["messages"] if m.type == "ai"]
        self.assertTrue(any("power" in m.content.lower() or "solution" in m.content.lower() for m in ai_messages))

if __name__ == "__main__":
    unittest.main()
