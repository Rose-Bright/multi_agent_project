from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from supervisors.customer_support.customer_support_supervisor import supervisor_graph
from supervisors.marketing.campaign_director_supervisor import campaign_graph
from workflows.search_literature_flow import workflow as literature_workflow
from agents.math_agent import MathAgent
from agents.writer_agent import WriterAgent
from utils.memory import ShortTermMemoryManager, LongTermMemoryManager
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

app = Flask(
    __name__,
    static_folder="static/frontend/build/static",
    static_url_path="/static"
)
CORS(app)

checkpointer = InMemorySaver()
global_short_term_manager = ShortTermMemoryManager()
global_long_term_manager = LongTermMemoryManager()
user_id = "user123"

@app.route('/')
def index():
    return send_from_directory('static/frontend/build', 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('static/frontend/build', path)

@app.route('/run_agent', methods=['POST'])
def run_agent():
    try:
        data = request.json
        agent_type = data.get('agent_type')
        messages = data.get('messages')
        message = data.get('message')
        thread_id = data.get('thread_id', user_id)

        if not messages:
            messages = [{"role": "user", "content": message}]

        if agent_type == 'customer_support':
            return run_customer_support(message)
        elif agent_type == 'marketing':
            return run_marketing(message)
        elif agent_type == 'literature':
            return run_literature(message)
        elif agent_type == 'math':
            return run_math_agent(messages, thread_id)
        elif agent_type == 'writer':
            return run_writer_agent(messages, thread_id)
        else:
            return jsonify({'error': 'Invalid agent type'}), 400
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'Backend exception: {str(e)}'}), 500

def run_customer_support(msg):
    state = MessagesState(messages=[HumanMessage(content=msg)])
    result = supervisor_graph.invoke(state)
    ai_messages = [m for m in result["messages"] if m.type == "ai"]
    return jsonify({'response': ai_messages[-1].content if ai_messages else "No AI response."})

def run_marketing(msg):
    state = MessagesState(messages=[HumanMessage(content=msg)])
    result = campaign_graph.invoke(state)
    ai_messages = [m for m in result["messages"] if m.type == "ai"]
    return jsonify({'response': ai_messages[-1].content if ai_messages else "No AI response."})

def run_literature(msg):
    messages = [HumanMessage(content=msg)]
    result = literature_workflow(messages)
    return jsonify({'response': result})

def run_math_agent(messages, thread_id):
    agent = MathAgent(
        short_term_memory=checkpointer,
        long_term_memory=global_long_term_manager.get_store(),
        user_id=user_id
    )

    latest_input = messages[-1]["content"]
    # Convert to LangChain message objects
    chat_history = []
    for m in messages:
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            from langchain_core.messages import AIMessage
            chat_history.append(AIMessage(content=m["content"]))

    result = agent.agent_executor.invoke(
        {"input": latest_input, "chat_history": chat_history},
        config={"configurable": {"thread_id": thread_id}}
    )
    return jsonify({'response': result["output"]})

def run_writer_agent(messages, thread_id):
    agent = WriterAgent(
        short_term_memory=checkpointer,
        long_term_memory=global_long_term_manager.get_store(),
        user_id=user_id
    )

    latest_input = messages[-1]["content"]
    # Convert to LangChain message objects
    chat_history = []
    for m in messages:
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            from langchain_core.messages import AIMessage
            chat_history.append(AIMessage(content=m["content"]))

    result = agent.agent_executor.invoke(
        {"input": latest_input, "chat_history": chat_history},
        config={"configurable": {"thread_id": thread_id}}
    )
    return jsonify({'response': result["output"]})

if __name__ == '__main__':
    app.run(debug=False)
