from agents.simple_agent import SimpleAgent

if __name__ == "__main__":
    agent = SimpleAgent()
    
    # Test cases
    # Test the calculator tool
    print(agent.choose_tool('What is 4 * 2?'))  # Should use the calculator tool and return '4'
    
    # Test the summarizer tool
    print(agent.choose_tool('Summarize this text: This is a long text. It has multiple sentences.'))  # Should use the summarizer tool and return 'This is a long text.'
    
    # Test a case where no tool is found
    # Here will test to translate a text to French which is not implemented in the agent
    # Will keep this as a test case to check if the agent returns 'Tool not found'
    print(agent.choose_tool('Translate this text to French: Hello, how are you?'))  # Should not find a tool and return 'Tool not found'