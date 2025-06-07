from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

@tool
def find_research_articles(topic: str) -> str:
    """Search for research articles based on the given topic."""
    return f"Found articles on '{topic}' from PubMed and arXiv."

@tool(return_direct=True)
def transfer_to_extract_agent() -> str:
    """Transfer task to extract data agent."""
    return "Switching to extract data agent."

search_tools = [find_research_articles, transfer_to_extract_agent]

search_agent = create_react_agent(
    model,
    search_tools,
    prompt=(
        "You are a literature search expert. Your job is to find relevant research articles based on a topic. "
        "Once articles are found, call 'extract_data_agent' to continue processing."
    ),
)
