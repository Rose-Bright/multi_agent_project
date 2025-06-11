import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage

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

writer_tools = [summarize_text, write_paragraph]

# ------------------ Agent ------------------

class WriterAgent:
    def __init__(self, short_term_memory, long_term_memory, user_id):
        """Initializes the WriterAgent with tools and memory."""
        self.user_id = user_id
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a technical writer agent. "
             "You can ONLY use the following tools: summarize_text and write_paragraph. "
             "If the user's request is not about summarizing text or writing a paragraph about a topic, "
             "you MUST respond with exactly: 'Tool not found'. "
             "Do NOT try to be helpful outside your tools. "
             "If you are unsure, respond with 'Tool not found'."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        self.agent = create_tool_calling_agent(self.llm, writer_tools, self.prompt)

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=writer_tools,
            verbose=True,
            checkpointer=self.short_term_memory,  # short-term memory for session/thread
            store=self.long_term_memory           # long-term memory for persistent data
        )

    def choose_tool(self, messages, config=None) -> str:
        if config is None:
            config = {}

        # Convert list of dicts to list of LangChain messages
        langchain_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))

        # Add to config
        config["configurable"]["chat_history"] = langchain_messages

        latest_input = messages[-1]["content"]
        response = self.agent_executor.invoke({"input": latest_input}, config=config)
        return response["output"]
