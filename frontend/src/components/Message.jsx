import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Copy, Check, Share2, RefreshCw, MoreHorizontal } from 'lucide-react';
import CodeBlock from './CodeBlock';

export default function Message({
  role,
  content,
  chartHtml,
  chartTitle,
  chartCaption,
  isLoading,
}) {
  const isUser = role === 'user';
  const [copied, setCopied] = useState(false);

  async function copyAll() {
    try {
      await navigator.clipboard.writeText(content || '');
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {}
  }

  if (isUser) {
    return (
      <div className="msg msg-user">
        <div className="msg-user-bubble">{content}</div>
      </div>
    );
  }

  const showActions = !isLoading && content && content.length > 0;

  return (
    <div className="msg msg-agent">
      <div className="msg-text">
        {isLoading && !content ? (
          <div className="loading-dots" aria-label="thinking">
            <span></span><span></span><span></span>
          </div>
        ) : (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              code({ inline, className, children }) {
                const match = /language-(\w+)/.exec(className || '');
                if (!inline && match) {
                  return (
                    <CodeBlock
                      language={match[1]}
                      code={String(children).replace(/\n$/, '')}
                    />
                  );
                }
                return <code className="inline-code">{children}</code>;
              },
            }}
          >
            {content || ''}
          </ReactMarkdown>
        )}

        {chartHtml && (
          <div className="msg-chart">
            <iframe
              srcDoc={chartHtml}
              title={chartTitle || 'Chart'}
              className="msg-chart-frame"
              sandbox="allow-scripts allow-same-origin"
            />
          </div>
        )}
      </div>
      {showActions && (
        <div className="msg-actions">
          <button
            type="button"
            className="msg-action"
            onClick={copyAll}
            aria-label={copied ? 'Copied' : 'Copy'}
            title={copied ? 'Copied' : 'Copy'}
          >
            {copied ? <Check size={15} strokeWidth={2} /> : <Copy size={15} strokeWidth={1.75} />}
          </button>
          <button
            type="button"
            className="msg-action"
            aria-label="Share"
            title="Share"
          >
            <Share2 size={15} strokeWidth={1.75} />
          </button>
          <button
            type="button"
            className="msg-action"
            aria-label="Regenerate"
            title="Regenerate"
          >
            <RefreshCw size={15} strokeWidth={1.75} />
          </button>
          <button
            type="button"
            className="msg-action"
            aria-label="More"
            title="More"
          >
            <MoreHorizontal size={15} strokeWidth={1.75} />
          </button>
        </div>
      )}
    </div>
  );
}
