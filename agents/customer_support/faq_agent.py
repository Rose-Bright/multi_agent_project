from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
import os
import logging
from dotenv import load_dotenv

load_dotenv()

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_FAQ_DATA = [
    {"question": "What is your return policy?", "answer": "You can return any item within 30 days of purchase."},
    {"question": "How do I reset my password?", "answer": "Click on 'Forgot password' at login and follow the instructions."},
    {"question": "Do you offer international shipping?", "answer": "Yes, we ship to most countries worldwide."}
]

@tool
def get_faq_answer(question: str) -> str:
    """Retrieve an answer to a frequently asked question."""
    for faq in _FAQ_DATA:
        if question.lower() in faq["question"].lower():
            return faq["answer"]
    return "Sorry, I couldn't find an answer to that question."

@tool
def list_all_faq_questions(_: str = "") -> str:
    """List all available FAQ questions."""
    return "\n".join(f"- {faq['question']}" for faq in _FAQ_DATA)

faq_tools = [get_faq_answer, list_all_faq_questions]

class FAQAgent:
    def __init__(self):
        """Initializes the FAQAgent with available tools."""
        self.llm = ChatVertexAI(
            model_name="gemini-2.5-flash",
            project=project_id,
            location=location,
            temperature=0.0,
                max_output_tokens=1024,
        )

        system_message = SystemMessage(content=(
             "You are a FAQ support agent. "
             "You can ONLY use the following tools: get_faq_answer and list_all_faq_questions. "
             "If the user's request is not about FAQs, "
             "you MUST respond with exactly: 'Tool not found'. "
             "Do NOT try to be helpful outside your tools. "
            "If you are unsure, respond with 'Tool not found'."
        ))

        self.prompt = ChatPromptTemplate.from_messages([
            system_message,
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])

        self.agent = create_tool_calling_agent(self.llm, faq_tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=faq_tools, verbose=True)
        logger.info("FAQAgent initialized with tools: %s", [tool.name for tool in faq_tools])

    def choose_tool(self, question: str) -> str:
        """
        Chooses the appropriate tool based on the input question using LangChain and Vertex AI.

        Args:
            question (str): The input question to analyze.

        Returns:
            str: The result from the chosen tool.
        """
        response = self.agent_executor.invoke({"input": question})
        return response["output"]
