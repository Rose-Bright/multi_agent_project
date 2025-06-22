import ast
import operator
import logging
import os
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Vertex AI configuration
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
# Define custom tools
@tool
def evaluate_expression(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        result = safe_eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def safe_eval(expression: str) -> float:
    """Safely evaluate a mathematical expression using ast."""
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg
    }

    def eval_node(node):
        if isinstance(node, ast.Num):  # <number>
            return node.n
        elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
            return allowed_operators[type(node.op)](eval_node(node.left), eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
            return allowed_operators[type(node.op)](eval_node(node.operand))
        else:
            raise TypeError(node)

    node = ast.parse(expression, mode='eval').body
    return eval_node(node)

@tool
def summarize_text(text: str) -> str:
    """Summarize the given text by returning the first sentence."""
    sentences = text.split('.')
    return sentences[0] + '.' if len(sentences) > 1 else text

tools = [evaluate_expression, summarize_text]

class SimpleAgent:
    """A simple agent with tools for calculation and summarization."""

    def __init__(self):
        """Initializes the SimpleAgent with available tools."""
        self.llm = ChatVertexAI(
            model_name="gemini-1.5-flash",
            project=project_id,
            location=location,
            temperature=0.0,
            max_output_tokens=1024
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a world class technical documentation writer. "
            "You can do two things only : "
            "1. Evaluate a mathematical expression safely."
            "2. Summarize a given text by returning the first sentence. "
            "If the user's request is not about evaluating a mathematical expression or summarizing text, "
            "you MUST respond with exactly: 'Tool not found'. "
            "Do NOT try to be helpful outside your tools. "),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        self.agent = create_tool_calling_agent(self.llm, tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)
        logger.info("SimpleAgent initialized with tools: %s", [tool.name for tool in tools])

    def choose_tool(self, question: str) -> str:
        """
        Chooses the appropriate tool based on the input question using LangChain and Vertex AI.

        Args:
            question (str): The input question to analyze.

        Returns:
            str: The result from the chosen tool.
        """
        response = self.agent_executor.invoke({"input": question})
        logger.info("Response for question '%s': %s", question, response["output"])
        return response["output"]

