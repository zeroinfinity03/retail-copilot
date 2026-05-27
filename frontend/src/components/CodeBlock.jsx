import { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { monoNight } from '../lib/monoNight';

export default function CodeBlock({ language, code }) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {}
  }

  return (
    <div className="code-block">
      <div className="code-block-header">
        <span className="code-block-lang">{language || 'text'}</span>
        <button
          type="button"
          className={`code-block-copy ${copied ? 'is-copied' : ''}`}
          onClick={handleCopy}
          aria-label="Copy code"
        >
          {copied ? (
            <>
              <Check size={13} strokeWidth={2.5} />
              <span>Copied</span>
            </>
          ) : (
            <>
              <Copy size={13} strokeWidth={2} />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <div className="code-block-body">
        <SyntaxHighlighter
          style={monoNight}
          language={language || 'text'}
          PreTag="div"
          customStyle={{
            margin: 0,
            padding: '16px 18px',
            background: 'transparent',
            fontSize: '13px',
          }}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
