import logging
import os
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Vertex AI configuration
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

@tool
def summarize_text(text: str) -> str:
    """Summarize the given text by returning the first sentence."""
    sentences = text.split('.')
    return sentences[0] + '.' if len(sentences) > 1 else text

@tool
def write_paragraph(topic: str) -> str:
    """Write a short paragraph about the given topic."""
    llm = ChatVertexAI(
        model_name="gemini-2.5-flash",
        project=project_id,
        location=location,
        temperature=0.7,
        max_output_tokens=1024
    )
    prompt = f"Write a concise, informative paragraph in English about: {topic}"
    response = llm.invoke(prompt)
    return response.content

writer_tools = [summarize_text, write_paragraph]

class WriterAgent:
    def __init__(self):
        """Initializes the WriterAgent with available tools."""
        self.llm = ChatVertexAI(
            model_name="gemini-2.5-flash",
            project=project_id,
            location=location,
            temperature=0.0,
            max_output_tokens=1024
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a technical writer agent. "
             "You can ONLY use the following tools: summarize_text and write_paragraph. "
             "If the user's request is not about summarizing text or writing a paragraph about a topic, "
             "you MUST respond with exactly: 'Tool not found'. "
             "Do NOT try to be helpful outside your tools. "
             "If you are unsure, respond with 'Tool not found'."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        self.agent = create_tool_calling_agent(self.llm, writer_tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=writer_tools, verbose=True)
        logger.info("WriterAgent initialized with tools: %s", [tool.name for tool in writer_tools])

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