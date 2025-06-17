import React, { useEffect, useRef, useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeMathjax from 'rehype-mathjax';

const agentOptions = [
  { value: 'customer_support', label: 'Customer Support' },
  { value: 'marketing', label: 'Marketing' },
  { value: 'literature', label: 'Literature Search' },
  { value: 'math', label: 'Math Agent' },
  { value: 'writer', label: 'Writer Agent' },
];

export default function App() {
  const [agent, setAgent] = useState(agentOptions[0].value);
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null); // Scroll target

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    const newEntry = { role: 'user', content: message };
    try {
      const res = await fetch('/run_agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agent_type: agent,
          message,
          messages: [...chatHistory, newEntry],
        }),
      });
      const data = await res.json();
      const aiEntry = { role: 'assistant', content: data.response || 'No response.' };
      setChatHistory([...chatHistory, newEntry, aiEntry]);
      setMessage('');
    } catch {
      setChatHistory([...chatHistory, newEntry, { role: 'system', content: 'Error connecting to backend.' }]);
    }
    setLoading(false);
  };

  return (
    <div className="container mt-5">
      <h1 className="mb-4">Multi-Agent System</h1>

      <form onSubmit={handleSubmit} className="mb-3">
        <div className="mb-3">
          <label htmlFor="agent-select" className="form-label">Choose an agent:</label>
          <select
            id="agent-select"
            className="form-select"
            value={agent}
            onChange={e => setAgent(e.target.value)}
          >
            {agentOptions.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        <div className="mb-3">
          <label htmlFor="user-message" className="form-label">Enter your message:</label>
          <textarea
            id="user-message"
            className="form-control"
            rows={3}
            value={message}
            onChange={e => setMessage(e.target.value)}
            required
          />
        </div>

        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? 'Processing...' : 'Submit'}
        </button>
      </form>

      {/* Chat style */}
      <div className="border rounded p-3 mb-3" style={{ maxHeight: "400px", overflowY: "auto", backgroundColor: "#f8f9fa" }}>
        {chatHistory.map((msg, index) => (
          <div
            key={index}
            className={`d-flex ${msg.role === 'user' ? 'justify-content-end' : 'justify-content-start'} mb-2`}
          >
            <div className={`p-2 rounded ${msg.role === 'user' ? 'bg-primary text-white' : 'bg-light text-dark'}`} style={{ maxWidth: "75%" }}>
              <strong>
                {msg.role === 'system' ? '⚠️ System' : msg.role === 'user' ? 'You' : 'Agent'}:
              </strong>
              <div>
                <ReactMarkdown
                  children={msg.content}
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeMathjax]}
                />
              </div>
            </div>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>
    </div>
  );
}
