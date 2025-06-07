from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

@tool
def generate_hypothesis(analysis: str) -> str:
    """Generate hypotheses based on analysis results."""
    return f"Hypothesis: Based on {analysis}, we propose..."

hypothesis_tools = [generate_hypothesis]

hypothesis_agent = create_react_agent(
    model,
    hypothesis_tools,
    prompt=(
        "You are a hypothesis generation agent. Use analysis results to formulate new research hypotheses. "
        "You are the final step."
    ),
)
