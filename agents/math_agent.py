import ast
import operator
import logging
import os
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Vertex AI configuration
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

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
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return allowed_operators[type(node.op)](eval_node(node.left), eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            return allowed_operators[type(node.op)](eval_node(node.operand))
        else:
            raise TypeError(node)
    node = ast.parse(expression, mode='eval').body
    return eval_node(node)

math_tools = [evaluate_expression]

class MathAgent:
    def __init__(self):
        """Initializes the MathAgent with available tools."""
        self.llm = ChatVertexAI(
            model_name="gemini-2.5-flash",
            project=project_id,
            location=location,
            temperature=0.0,
            max_output_tokens=1024
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a math expert. "
             "You can ONLY use the following tool: evaluate_expression. "
             "If the user's request is not about evaluating a mathematical expression, "
             "you MUST respond with exactly: 'Tool not found'. "
             "Do NOT try to be helpful outside your tool. "
             "If you are unsure, respond with 'Tool not found'."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        self.agent = create_tool_calling_agent(self.llm, math_tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=math_tools, verbose=True)
        logger.info("MathAgent initialized with tools: %s", [tool.name for tool in math_tools])

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