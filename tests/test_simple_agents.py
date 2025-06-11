import unittest
from agents.math_agent import MathAgent
from agents.writer_agent import WriterAgent

class TestSimpleAgents(unittest.TestCase):
    def test_math_agent(self):
        agent = MathAgent()
        result = agent.choose_tool("2 + 2")
        self.assertIn("4", result)

    def test_writer_agent_summarize(self):
        agent = WriterAgent()
        text = "Python is a programming language. It is widely used."
        result = agent.choose_tool(f"Summarize: {text}")
        self.assertIn("Python is a programming language", result)

    def test_writer_agent_paragraph(self):
        agent = WriterAgent()
        topic = "Artificial Intelligence"
        result = agent.choose_tool(f"Write a paragraph about {topic}")
        self.assertTrue("Artificial Intelligence" in result or len(result) > 10)

if __name__ == "__main__":
    unittest.main()
