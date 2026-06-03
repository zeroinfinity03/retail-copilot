const USE_MOCK = false;

/**
 * Streams the response from /api/chat via Server-Sent Events.
 *
 * Calls back as events arrive:
 *   onToken(chunk)           — incremental synthesizer token
 *   onComplete({ charts }) — final event with an array of { chart_html, chart_title, chart_caption }
 *   onError(error)           — fatal error
 *
 * Returns a Promise that resolves when the stream finishes.
 */
export async function sendAgentRequest(messages, { onToken, onComplete, onError }) {
  if (USE_MOCK) {
    await mockStream(messages, onToken, onComplete);
    return;
  }

  let resp;
  try {
    resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages }),
    });
  } catch (err) {
    onError(err);
    return;
  }

  if (!resp.ok) {
    onError(new Error(`HTTP ${resp.status}`));
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by a blank line ("\n\n")
      let separator;
      while ((separator = buffer.indexOf('\n\n')) !== -1) {
        const rawEvent = buffer.slice(0, separator);
        buffer = buffer.slice(separator + 2);

        const line = rawEvent.trim();
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;

        let event;
        try {
          event = JSON.parse(payload);
        } catch {
          continue;
        }

        if (event.type === 'text' && typeof event.content === 'string') {
          onToken(event.content);
        } else if (event.type === 'complete') {
          onComplete({ charts: Array.isArray(event.charts) ? event.charts : [] });
        } else if (event.type === 'error') {
          onError(new Error(event.error || 'Stream error'));
          return;
        }
      }
    }
  } catch (err) {
    onError(err);
  }
}

async function mockStream(messages, onToken, onComplete) {
  const last = messages[messages.length - 1]?.content || '';
  const tokens = [
    'Running ', 'in ', 'mock ', 'mode.\n\n',
    `You asked: *"${last}"*\n\n`,
    'Flip ', 'USE_MOCK ', 'back ', 'to ', 'true ', 'in ', 'chatStream.js ', 'to ', 'use ', 'this ', 'stub.',
  ];
  for (const t of tokens) {
    await new Promise((r) => setTimeout(r, 40));
    onToken(t);
  }
  onComplete({ charts: [] });
}
