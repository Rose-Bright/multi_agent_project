import os

import vertexai
from vertexai import agent_engines
from dotenv import load_dotenv

# Import your supervisor agent
from supervisors.customer_support.customer_support_supervisor import agent as customer_support_supervisor_agent

load_dotenv()

vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION"),
    staging_bucket="gs://" + os.getenv("GOOGLE_CLOUD_PROJECT")+"-bucket",
)

# List of (display_name, agent_object)
AGENTS = [
    ("Customer Support Supervisor Agent", customer_support_supervisor_agent),
    # Add other agents as needed
]

for display_name, agent in AGENTS:
    remote_app = agent_engines.create(
        display_name=display_name,
        agent_engine=agent,
        requirements=[
            "fastapi==0.115.13",
            "google-cloud-aiplatform[adk,agent_engines,langgraph]==1.97.0",
            "google-adk[vertexai]==1.3.0",
            "langchain-community==0.3.25",
            "langchain-openai==0.3.8",
            "python-dotenv==1.0.1",
            "openai==1.88.0",
        ],
        extra_packages=[
            "./agents/customer_support",
            "./supervisors/customer_support",
        ],
        env_vars={
            "OPENAI_API_KEY": {
                "secret": "OPENAI_API_KEY",        # Replace with your Secret Manager secret ID
                "version": "latest",  # Replace with the version (e.g., "latest")
            },
            # Add other environment variables or secrets as needed
        }
    )
    print(f"Deployed {display_name}")
