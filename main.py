from logging_config import setup_logging
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from supervisors.customer_support.customer_support_supervisor import supervisor_graph
from supervisors.marketing.campaign_director_supervisor import campaign_graph
from workflows.search_literature_flow import workflow as literature_workflow
from agents.math_agent import MathAgent
from agents.writer_agent import WriterAgent
from utils.memory import MemoryManager
from utils.chat_utils import handle_agent_chat
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
setup_logging()
app = Flask(
    __name__,
    static_folder="static/frontend/build/static",
    static_url_path="/static"
)
CORS(app)

memory_manager = MemoryManager()
user_id = "123"

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
    agent = MathAgent(memory_manager, user_id)
    output = handle_agent_chat(agent, messages, thread_id, memory_manager)
    return jsonify({'response': output})

def run_writer_agent(messages, thread_id):
    agent = WriterAgent(memory_manager, user_id)
    output = handle_agent_chat(agent, messages, thread_id, memory_manager)
    return jsonify({'response': output})

@app.route('/get_memory', methods=['GET'])
def get_memory():
    thread_id = request.args.get('thread_id', user_id)
    query = request.args.get('query', '')
    chat_history = memory_manager.get_latest_chat_history(thread_id)
    short_term_list = [
        {
            'text': getattr(m, 'content', ''),
            'role': getattr(m, 'role', 'user' if getattr(m, 'type', '') == 'human' else 'assistant'),
            'agent': getattr(m, 'agent', None),  # Add this if your message object has agent info
            'timestamp': getattr(m, 'ts', None)  # Add this if your message object has timestamp info
        }
        for m in chat_history
    ] if chat_history else []

    if query:
        long_term_memory = memory_manager.search_long_term_memory(user_id, query)
    else:
        long_term_memory = memory_manager.get_all_long_term_memory(user_id)

    long_term_list = [
        {
            'text': m['text'],
            'role': m['metadata'].get('role'),
            'agent': m['metadata'].get('agent'),
            'ts': m['metadata'].get('timestamp')
        }
        for m in long_term_memory
    ] if long_term_memory else []

    return jsonify({
        'short_term_memory': short_term_list,
        'long_term_memory': long_term_list
    })

if __name__ == '__main__':
    app.run(debug=False)
