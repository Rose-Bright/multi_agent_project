import ast
import operator
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from utils.chat_utils import prepare_agent_input
from tools.memory_tools import retrieve_memory

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ------------------ Tool ------------------

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
        if isinstance(node, ast.Num):  # For Python <3.8
            return node.n
        elif isinstance(node, ast.Constant):  # For Python >=3.8
            return node.value
        elif isinstance(node, ast.BinOp):
            return allowed_operators[type(node.op)](eval_node(node.left), eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            return allowed_operators[type(node.op)](eval_node(node.operand))
        else:
            raise TypeError(node)

    node = ast.parse(expression, mode='eval').body
    return eval_node(node)

math_tools = [evaluate_expression, retrieve_memory]

# ------------------ Agent ------------------

class MathAgent:
    def __init__(self, memory_manager, user_id):
        self.memory = memory_manager
        self.user_id = user_id
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a math expert. "
             "You MUST use ONLY the tools: evaluate_expression and retrieve_memory to answer ANY request. "
             "NEVER answer a math question directly or do calculations yourself, even after retrieving a value from memory. "
             "If you retrieve a value from memory and need to use it in a calculation, you MUST call evaluate_expression with the correct expression. "
             "If you do not use the tool, your answer could be invalid and will have to be rejected in case you hallucinate. "
             "If the user's request refers to a previous answer or question, use both the chat history and any relevant past knowledge provided to resolve what the user means. "
             "ALWAYS format your answers using Markdown. For math, use LaTeX in double dollar signs ($$...$$)."
             "For example, reply: The result of the expression $$2 \\times 2$$ is $$4$$. "
             r"for example, do NOT reply: 'The result of multiplying the last math result \(4\) by \(10\) is \(40\).' "
             "If you cannot resolve the request as a math expression, respond with: 'Tool not found'."
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        self.agent = create_tool_calling_agent(self.llm, math_tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=math_tools,
            verbose=True,
            checkpointer=self.memory.get_short_term(),
            store=self.memory.get_long_term(),
        )

    def choose_tool(self, messages, config=None) -> str:
        return prepare_agent_input(self, self.agent, messages, self.memory, self.user_id, config)
