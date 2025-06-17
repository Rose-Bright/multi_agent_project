import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from utils.chat_utils import prepare_agent_input
from tools.memory_tools import retrieve_memory
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ------------------ Tools ------------------

@tool
def summarize_text(text: str) -> str:
    """Summarize the given text by returning the first sentence."""
    sentences = text.split('.')
    return sentences[0] + '.' if len(sentences) > 1 else text

@tool
def write_paragraph(topic: str) -> str:
    """Write a short paragraph about the given topic."""
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
    prompt = f"Write a concise, informative paragraph in English about: {topic}"
    response = llm.invoke(prompt)
    return response.content

writer_tools = [summarize_text, write_paragraph, retrieve_memory]

# ------------------ Agent ------------------

class WriterAgent:
    def __init__(self, memory_manager, user_id):
        self.memory = memory_manager
        self.user_id = user_id
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a technical writer agent. "
             "You MUST use ONLY the tools: summarize_text, write_paragraph, and retrieve_memory to answer ANY request. "
             "NEVER answer a request directly or try to be helpful outside your tools, even after retrieving information from memory. "
             "If you retrieve information from memory and need to summarize it or write a paragraph, you MUST call summarize_text or write_paragraph with the appropriate input. "
             "If the user's request is about previous answers, facts, or context, you MUST use the retrieve_memory tool. "
             "If the user's request is not about summarizing text, writing a paragraph, or retrieving memory, "
             "you MUST respond with exactly: 'Tool not found'. "
             "Any other answer will be rejected as this is not part of your task. "
             "Do NOT attempt to answer or explain anything else. "
             "For example, do NOT reply: 'Sorry, I cannot help with that.' or 'I am a technical writer agent.' "
             "If you are unsure, respond with 'Tool not found'."
             "After using a tool, you should return the result of that tool's execution. "
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        self.agent = create_tool_calling_agent(self.llm, writer_tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=writer_tools,
            verbose=True,
            checkpointer=self.memory.get_short_term(),
            store=self.memory.get_long_term(),
        )

    def choose_tool(self, messages, config=None) -> str:
        return prepare_agent_input(self, self.agent, messages, self.memory, self.user_id, config)
