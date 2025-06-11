import unittest
from supervisors.marketing.campaign_director_supervisor import campaign_graph
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage

class TestMarketingSupervisor(unittest.TestCase):
    def test_launch_campaign(self):
        msg = "Launch a new marketing campaign for our summer product line."
        state = MessagesState(messages=[HumanMessage(content=msg)])
        result = campaign_graph.invoke(state)
        ai_messages = [m for m in result["messages"] if m.type == "ai"]
        self.assertTrue(any("campaign" in m.content.lower() for m in ai_messages))

    def test_analyze_performance(self):
        msg = "Analyze the last campaign's performance."
        state = MessagesState(messages=[HumanMessage(content=msg)])
        result = campaign_graph.invoke(state)
        ai_messages = [m for m in result["messages"] if m.type == "ai"]
        self.assertTrue(any("analysis" in m.content.lower() or "performance" in m.content.lower() for m in ai_messages))

    def test_deploy_creatives(self):
        msg = "Deploy the new ad creatives to all channels."
        state = MessagesState(messages=[HumanMessage(content=msg)])
        result = campaign_graph.invoke(state)
        ai_messages = [m for m in result["messages"] if m.type == "ai"]
        self.assertTrue(any("deploy" in m.content.lower() or "deployment" in m.content.lower() for m in ai_messages))

if __name__ == "__main__":
    unittest.main()
