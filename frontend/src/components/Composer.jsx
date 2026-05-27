import { useState, useRef, useEffect } from 'react';
import { Plus, Globe, ArrowUp } from 'lucide-react';

export default function Composer({ onSend, disabled }) {
  const [text, setText] = useState('');
  const [webOn, setWebOn] = useState(false);
  const taRef = useRef(null);

  useEffect(() => {
    taRef.current?.focus();
  }, []);

  function submit(e) {
    e?.preventDefault?.();
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed, { web: webOn });
    setText('');
    if (taRef.current) taRef.current.style.height = 'auto';
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function onChange(e) {
    setText(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
  }

  const hasText = text.trim().length > 0;

  return (
    <div className="composer-wrap">
      <form className="composer" onSubmit={submit}>
        <div className="composer-left">
          <button
            type="button"
            className="icon-btn"
            title="Attach"
            aria-label="Attach"
            tabIndex={-1}
          >
            <Plus size={20} strokeWidth={2} />
          </button>
          <button
            type="button"
            className={`icon-btn ${webOn ? 'is-on' : ''}`}
            title={webOn ? 'Web search: on' : 'Web search: off'}
            aria-label="Toggle web search"
            aria-pressed={webOn}
            onClick={() => setWebOn((v) => !v)}
          >
            <Globe size={18} strokeWidth={1.75} />
          </button>
        </div>

        <textarea
          ref={taRef}
          value={text}
          onChange={onChange}
          onKeyDown={onKeyDown}
          placeholder="Ask anything"
          rows={1}
          disabled={disabled}
          spellCheck="false"
        />

        <button
          type="submit"
          className="send-btn"
          disabled={!hasText || disabled}
          title="Send"
          aria-label="Send"
        >
          <ArrowUp size={17} strokeWidth={2.5} />
        </button>
      </form>
    </div>
  );
}
