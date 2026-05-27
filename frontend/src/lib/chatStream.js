const USE_MOCK = false;

/**
 * Sends the message list to /api/chat and returns the parsed response.
 * Shape: { text: string, chart_html: string | null, chart_title?: string, chart_caption?: string }
 */
export async function sendAgentRequest(messages) {
  if (USE_MOCK) {
    return mockResponse(messages);
  }
  return realRequest(messages);
}

async function realRequest(messages) {
  const resp = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

async function mockResponse(messages) {
  // Tiny delay so the UI shows a loading state
  await new Promise((r) => setTimeout(r, 600));
  const last = messages[messages.length - 1]?.content || '';
  return {
    text: `Running in mock mode.\n\nYou asked: *"${last}"*\n\nFlip USE_MOCK back to true in chatStream.js to use this stub.`,
    chart_html: null,
  };
}
