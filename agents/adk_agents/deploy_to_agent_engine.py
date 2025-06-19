import os

import vertexai
from vertexai import agent_engines
from dotenv import load_dotenv

# Use full package path for agent imports
from agents.adk_agents.wikipedia_langchain_tool_agent.wikipedia_langchain_tool_agent import wikipedia_langchain_tool_agent

load_dotenv()

vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION"),
    staging_bucket="gs://" + os.getenv("GOOGLE_CLOUD_PROJECT")+"-bucket",
)

# List of (display_name, agent_object)
AGENTS = [
    #("Journaling Function Tool Agent", journaling_function_tool_agent),
    #("Transcript Summarization Agent", transcript_summarization_agent),
    #("Travel Planning Parent And Subagents", travel_planning_parent_and_subagents),
    #("VertexAI Scientific Search Agent", vertexai_scientific_search_agent),
    ("Wikipedia Langchain Tool Agent", wikipedia_langchain_tool_agent),
    # Add more as needed
]

for display_name, agent in AGENTS:
    remote_app = agent_engines.create(
        display_name=display_name,
        agent_engine=agent,
        requirements=[
            "fastapi==0.115.13",
            "google-cloud-aiplatform[adk,agent_engines]==1.97.0",
            "a2a-sdk==0.2.8",
            "wikipedia==1.4.0",
            "dateparser==1.2.1",
            "langchain-community==0.3.25",
        ],
        extra_packages=["./agents"]
    )
    print(f"Deployed {display_name}")
