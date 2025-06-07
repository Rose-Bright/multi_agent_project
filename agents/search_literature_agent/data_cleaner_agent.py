from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

@tool
def clean_extracted_data(data: str) -> str:
    """Cleans extracted data from noise or irrelevant content."""
    return f"Cleaned data: {data[:50]}..."

@tool(return_direct=True)
def transfer_to_analyze_agent() -> str:
    """Transfer task to analyze data agent."""
    return "Switching to analyze data agent."

cleaner_tools = [clean_extracted_data, transfer_to_analyze_agent]

cleaner_agent = create_react_agent(
    model,
    cleaner_tools,
    prompt=(
        "You are a data cleaner agent. Clean up extracted data to prepare it for analysis. "
        "After cleaning, call the analyzer agent."
    ),
)
