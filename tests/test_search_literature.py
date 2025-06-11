import unittest
from workflows.search_literature_flow import workflow
from langchain_core.messages import HumanMessage

class TestSearchLiteratureWorkflow(unittest.TestCase):
    def test_search_literature(self):
        messages = [HumanMessage(content="I'd like to search recent research on CRISPR in cancer. Clean and analyze the data, then generate a hypothesis.")]
        result = workflow(messages)
        # Check that the workflow returns a list of messages and at least one message contains 'CRISPR' or 'research'
        self.assertTrue(any("CRISPR" in m.content or "research" in m.content for m in result))

if __name__ == "__main__":
    unittest.main()
