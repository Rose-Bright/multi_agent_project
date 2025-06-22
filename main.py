from agents.math_agent import MathAgent
from agents.writer_agent import WriterAgent

if __name__ == "__main__":
    math_agent = MathAgent()
    writer_agent = WriterAgent()

    # Test cases
    # Test the calculator tool
    print("MathAgent result:", math_agent.choose_tool('What is 4 * 2?'))  # Should use the calculator tool and return '4'

    # Test the summarizer tool
    print("WriterAgent result:", writer_agent.choose_tool('Summarize this text: This is a long text. It has multiple sentences.'))  # Should use the summarizer tool and return 'This is a long text.'

    # Test to write a paragraph
    print("WriterAgent result:", writer_agent.choose_tool('Write a paragraph about AI.'))  # Should use the write paragraph tool and return a placeholder paragraph

    # Test a case where no tool is found
    # Here will test to translate a text to French which is not implemented in the agent
    # Will keep this as a test case to check if the agent returns 'Tool not found'
    print(math_agent.choose_tool('Translate this text to French: Hello, how are you?'))  # Should not find a tool and return 'Tool not found'
    print(writer_agent.choose_tool('What is 4 * 2?'))  # Should not find a tool and return 'Tool not found'
