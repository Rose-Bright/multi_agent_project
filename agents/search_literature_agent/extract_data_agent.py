from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

@tool
def extract_figures_and_results(article_summary: str) -> str:
    """Extracts figures and key results from article summary."""
    return f"Extracted tables and key results from: {article_summary}"

@tool(return_direct=True)
def transfer_to_cleaner_agent() -> str:
    """Transfer task to data cleaner agent."""
    return "Switching to data cleaner agent."

extract_tools = [extract_figures_and_results, transfer_to_cleaner_agent]

extract_agent = create_react_agent(
    model,
    extract_tools,
    prompt=(
        "You are a data extraction agent. Given article summaries, you extract key figures, tables, and insights. "
        "Transfer to the cleaner agent once extraction is complete."
    ),
)
