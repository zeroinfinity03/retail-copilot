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

    // Helper: update the in-flight assistant message (always the last one)
    const updateAssistant = (patch) => {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        next[next.length - 1] = { ...last, ...patch };
        return next;
      });
    };

    await sendAgentRequest([...messages, userMsg], {
      onToken: (chunk) => {
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          next[next.length - 1] = {
            ...last,
            content: (last.content || '') + chunk,
            loading: false,
          };
          return next;
        });
      },
      onComplete: (chart) => {
        updateAssistant({
          chart_html: chart.chart_html,
          chart_title: chart.chart_title,
          chart_caption: chart.chart_caption,
          loading: false,
        });
      },
      onError: (err) => {
        updateAssistant({
          content: `**Error:** ${err.message || 'Something went wrong'}`,
          loading: false,
        });
      },
    });

    setIsLoading(false);
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
