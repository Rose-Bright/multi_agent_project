import unittest
from agents.math_agent import MathAgent
from agents.writer_agent import WriterAgent

class TestAgentMemory(unittest.TestCase):
    def test_math_agent_memory(self):
        agent = MathAgent()
        agent.choose_tool("3 * 3")
        # Should be in short-term memory
        short_term = agent.recall_short_term()
        self.assertTrue(any("3 * 3" in m["content"] for m in short_term if m["role"] == "user"))
        # Should be in long-term memory
        self.assertEqual(agent.recall_long_term("3 * 3"), "9")

    def test_writer_agent_memory(self):
        agent = WriterAgent()
        agent.choose_tool("Summarize: AI is the future. It will change everything.")
        short_term = agent.recall_short_term()
        self.assertTrue(any("Summarize" in m["content"] for m in short_term if m["role"] == "user"))
        # Long-term memory stores output for the question
        self.assertIsNotNone(agent.recall_long_term("Summarize: AI is the future. It will change everything."))

if __name__ == "__main__":
    unittest.main()
