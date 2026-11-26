import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import DecisionBadge from "../components/DecisionBadge";

export default function TryIt() {
  const [form, setForm] = useState({
    amount: 250,
    time: 50000,
    entity_id: "card_demo",
    merchant_id: "merch_demo",
    V14: -3.5,
    V12: -2.0,
    V17: 2.0,
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  function set(k, v) {
    setForm((f) => ({ ...f, [k]: v }));
  }

  async function submit(e) {
    e.preventDefault();
    setLoading(true);
    setErr("");
    try {
      const body = {
        amount: Number(form.amount),
        time: Number(form.time),
        entity_id: form.entity_id,
        merchant_id: form.merchant_id,
        features: { V14: Number(form.V14), V12: Number(form.V12), V17: Number(form.V17) },
      };
      const res = await api.score(body);
      setResult(res);
    } catch (ex) {
      setErr(String(ex.message || ex));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-5 animate-rise">
      <div>
        <h1 className="font-display text-4xl font-bold">Try It Yourself</h1>
        <p className="mt-2 text-slate-400">Score a custom transaction — ideal for live demos.</p>
      </div>
      <form className="panel grid gap-4 p-5 md:grid-cols-2" onSubmit={submit}>
        {[
          ["amount", "Amount"],
          ["time", "Time (seconds)"],
          ["entity_id", "Entity / card ID"],
          ["merchant_id", "Merchant ID"],
          ["V14", "Feature V14"],
          ["V12", "Feature V12"],
          ["V17", "Feature V17"],
        ].map(([k, label]) => (
          <label key={k} className="text-sm">
            {label}
            <input
              className="input mt-1"
              value={form[k]}
              onChange={(e) => set(k, e.target.value)}
              type={k.includes("id") ? "text" : "number"}
              step="any"
            />
          </label>
        ))}
        <div className="md:col-span-2">
          <button className="btn-primary" type="submit" disabled={loading}>
            {loading ? "Scoring…" : "Score transaction"}
          </button>
        </div>
      </form>
      {err && <div className="text-alert-400">{err}</div>}
      {result && (
        <div className="panel space-y-3 p-5">
          <div className="flex flex-wrap items-center gap-3">
            <DecisionBadge decision={result.decision} />
            <Link className="font-mono text-sm text-mint-400" to={`/tx/${result.transaction_id}`}>
              {result.transaction_id}
            </Link>
            <span className="font-mono text-sm">p={result.p_fraud.toFixed(4)}</span>
            <span className="font-mono text-sm text-slate-400">{result.latency_ms.toFixed(1)} ms</span>
          </div>
          <ul className="space-y-2 text-sm text-slate-300">
            {result.top_reasons.map((r) => (
              <li key={r.feature}>• {r.text}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
