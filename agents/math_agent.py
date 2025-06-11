import ast
import operator
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage

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

math_tools = [evaluate_expression]

# ------------------ Agent ------------------

class MathAgent:
    def __init__(self, short_term_memory, long_term_memory, user_id):
        """Initializes the MathAgent with tools and memory."""
        self.user_id = user_id
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are a math expert. "
            "You can ONLY use the following tool: evaluate_expression. "
            "Only call this tool once. "
            "Once you've received the result, return it immediately as the final answer. "
            "If the user's request is not a math expression, respond with: 'Tool not found'."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        self.agent = create_tool_calling_agent(self.llm, math_tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=math_tools,
            verbose=True,
            checkpointer=self.short_term_memory,
            store=self.long_term_memory,
        )

    def choose_tool(self, messages, config=None) -> str:
        if config is None:
            config = {}

        # Convert list of dicts to list of LangChain messages
        langchain_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))

        # Add to config
        config["configurable"]["chat_history"] = langchain_messages

        latest_input = messages[-1]["content"]
        response = self.agent_executor.invoke({"input": latest_input}, config=config)
        return response["output"]
