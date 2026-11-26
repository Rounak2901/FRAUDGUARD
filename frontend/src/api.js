const API = import.meta.env.VITE_API_URL || "";

async function req(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || res.statusText);
  }
  return res.json();
}

export const api = {
  health: () => req("/health"),
  metrics: () => req("/api/v1/metrics"),
  transactions: (params = {}) => {
    const q = new URLSearchParams(params).toString();
    return req(`/api/v1/transactions${q ? `?${q}` : ""}`);
  },
  transaction: (id) => req(`/api/v1/transactions/${id}`),
  score: (body) => req("/api/v1/score", { method: "POST", body: JSON.stringify(body) }),
  feedback: (body) => req("/api/v1/feedback", { method: "POST", body: JSON.stringify(body) }),
  entity: (id) => req(`/api/v1/entities/${encodeURIComponent(id)}`),
  streamStart: (body) => req("/api/v1/stream/start", { method: "POST", body: JSON.stringify(body) }),
  streamStop: () => req("/api/v1/stream/stop", { method: "POST", body: "{}" }),
  costSimulate: (body) => req("/api/v1/cost/simulate", { method: "POST", body: JSON.stringify(body) }),
};

export function liveSocket(onMessage) {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const host = import.meta.env.VITE_WS_URL || `${proto}://${window.location.hostname}:8001`;
  const ws = new WebSocket(`${host}/ws/live`);
  ws.onmessage = (ev) => {
    try {
      onMessage(JSON.parse(ev.data));
    } catch {
      /* ignore */
    }
  };
  const ping = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.send("ping");
  }, 20000);
  ws.onclose = () => clearInterval(ping);
  return ws;
}
