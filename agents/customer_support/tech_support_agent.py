from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_TECH_SUPPORT_DATA = [
    {"issue": "Cannot connect to Wi-Fi", "solution": "Restart your router and check your Wi-Fi password."},
    {"issue": "Computer won't turn on", "solution": "Check the power cable and try a different outlet."},
    {"issue": "Screen is flickering", "solution": "Update your graphics driver or check the cable connection."}
]

@tool
def get_tech_solution(issue: str) -> str:
    """Retrieve a solution to a common technical issue."""
    for item in _TECH_SUPPORT_DATA:
        if issue.lower() in item["issue"].lower():
            return item["solution"]
    return "Sorry, I couldn't find a solution for that issue."

@tool
def list_common_issues(_: str = "") -> str:
    """List all common technical issues."""
    return "\n".join(f"- {item['issue']}" for item in _TECH_SUPPORT_DATA)

tech_support_tools = [get_tech_solution, list_common_issues]

class TechSupportAgent:
    def __init__(self):
        """Initializes the TechSupportAgent with available tools."""
        self.llm = ChatVertexAI(
                model_name="gemini-2.5-flash",
                project=project_id,
                location=location,
                temperature=0.0,
                max_output_tokens=1024,
                convert_system_message_to_human=True)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a technical support agent. "
             "You can ONLY use the following tools: get_tech_solution and list_common_issues. "
             "If the user's request is not about technical support, "
             "you MUST respond with exactly: 'Tool not found'. "
             "Do NOT try to be helpful outside your tools. "
             "If you are unsure, respond with 'Tool not found'."),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        self.agent = create_tool_calling_agent(self.llm, tech_support_tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tech_support_tools, verbose=True)
        logger.info("TechSupportAgent initialized with tools: %s", [tool.name for tool in tech_support_tools])

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

# Example of how the fake tech support data could be stored in a JSON file:
# filepath: c:\projects\multi_agent_project\agents\customer_support\tech_support_data.json
"""
[
    {"issue": "Cannot connect to Wi-Fi", "solution": "Restart your router and check your Wi-Fi password."},
    {"issue": "Computer won't turn on", "solution": "Check the power cable and try a different outlet."},
    {"issue": "Screen is flickering", "solution": "Update your graphics driver or check the cable connection."}
]
"""
