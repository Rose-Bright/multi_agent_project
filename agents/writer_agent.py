from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@tool
def summarize_text(text: str) -> str:
    """Summarize the given text by returning the first sentence."""
    sentences = text.split('.')
    return sentences[0] + '.' if len(sentences) > 1 else text

@tool
def write_paragraph(topic: str) -> str:
    """Write a short paragraph about the given topic."""
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
    prompt = f"Write a concise, informative paragraph in English about: {topic}"
    response = llm.invoke(prompt)
    return response.content
writer_tools = [summarize_text, write_paragraph]

class WriterAgent:
    def __init__(self):
        """Initializes the WriterAgent with available tools."""
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
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

    def choose_tool(self, question: str) -> str:
        """
        Chooses the appropriate tool based on the input question using LangChain and OpenAI.

        Args:
            question (str): The input question to analyze.

        Returns:
            str: The result from the chosen tool.
        """
        response = self.agent_executor.invoke({"input": question})
        return response["output"]
