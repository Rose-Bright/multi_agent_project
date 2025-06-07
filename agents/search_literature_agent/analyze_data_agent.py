from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

@tool
def analyze_cleaned_data(data: str) -> str:
    """Analyze cleaned data to identify patterns, conclusions, or metrics."""
    return f"Analysis result: Significant increase in {data[:20]}"

@tool(return_direct=True)
def transfer_to_hypothesis_agent() -> str:
    """Transfer task to hypothesis agent."""
    return "Switching to hypothesis agent."

analyze_tools = [analyze_cleaned_data, transfer_to_hypothesis_agent]

analyze_agent = create_react_agent(
    model,
    analyze_tools,
    prompt=(
        "You are a data analyst. Take cleaned data and run basic analysis. "
        "Afterwards, transfer to the hypothesis agent."
    ),
)
