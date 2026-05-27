import { useState, useRef, useEffect } from 'react';
import Message from './components/Message';
import Composer from './components/Composer';
import { sendAgentRequest } from './lib/chatStream';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  async function handleSend(text) {
    const userMsg = { role: 'user', content: text };
    // Add user msg + a placeholder assistant msg
    setMessages((prev) => [
      ...prev,
      userMsg,
      { role: 'assistant', content: '', chart_html: null, loading: true },
    ]);
    setIsLoading(true);

    try {
      const data = await sendAgentRequest([...messages, userMsg]);
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = {
          role: 'assistant',
          content: data.text || '',
          chart_html: data.chart_html || null,
          chart_title: data.chart_title || null,
          chart_caption: data.chart_caption || null,
          loading: false,
        };
        return next;
      });
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = {
          role: 'assistant',
          content: `**Error:** ${err.message || 'Something went wrong'}`,
          loading: false,
        };
        return next;
      });
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="app">
      <main className="scroll" ref={scrollRef}>
        <div className="scroll-inner">
          <div className="thread">
            {messages.map((m, i) => (
              <Message
                key={i}
                role={m.role}
                content={m.content}
                chartHtml={m.chart_html}
                chartTitle={m.chart_title}
                chartCaption={m.chart_caption}
                isLoading={m.loading}
              />
            ))}
          </div>
        </div>
      </main>

      <Composer onSend={handleSend} disabled={isLoading} />
    </div>
  );
}
