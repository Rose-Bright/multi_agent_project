import logging
from agents.math_agent import MathAgent
from agents.writer_agent import WriterAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        math_agent = MathAgent()
        writer_agent = WriterAgent()

        # Test cases
        logger.info("Testing MathAgent...")
        # Test the calculator tool
        result = math_agent.choose_tool('What is 4 * 2?')
        print("MathAgent result:", result)

        logger.info("Testing WriterAgent summarize...")
        # Test the summarizer tool
        result = writer_agent.choose_tool('Summarize this text: This is a long text. It has multiple sentences.')
        print("WriterAgent summarize result:", result)

        logger.info("Testing WriterAgent paragraph...")
        # Test to write a paragraph
        result = writer_agent.choose_tool('Write a paragraph about AI.')
        print("WriterAgent paragraph result:", result)

        logger.info("Testing tool not found cases...")
        # Test cases where no tool is found
        result = math_agent.choose_tool('Translate this text to French: Hello, how are you?')
        print("MathAgent no tool result:", result)
        
        result = writer_agent.choose_tool('What is 4 * 2?')
        print("WriterAgent no tool result:", result)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")