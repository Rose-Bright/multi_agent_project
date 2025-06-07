import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@tool
def post_on_social_media(message: str, platform: str) -> str:
    """Post a message on a specified social media platform."""
    # Fake logic: return a confirmation message
    if not message or not platform:
        return "Message or platform missing."
    return f"Posted on {platform}: {message}"

social_tools = [post_on_social_media]

class SocialAgent:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a social media agent. You can ONLY use the following tool: post_on_social_media. If the user's request is not about posting on social media, you MUST respond with exactly: 'Tool not found'. Do NOT try to be helpful outside your tool. If you are unsure, respond with 'Tool not found'."),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        self.agent = create_tool_calling_agent(self.llm, social_tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=social_tools, verbose=True)

    def choose_tool(self, question: str) -> str:
        response = self.agent_executor.invoke({"input": question})
        return response["output"]
