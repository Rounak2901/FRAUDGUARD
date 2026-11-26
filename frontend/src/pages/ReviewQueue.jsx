import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import DecisionBadge from "../components/DecisionBadge";

export default function ReviewQueue() {
  const [items, setItems] = useState([]);
  const [msg, setMsg] = useState("");

  async function load() {
    const data = await api.transactions({ decision: "REVIEW", limit: 100 });
    const sorted = [...(data.items || [])].sort(
      (a, b) => Number(b.expected_loss_if_approved || 0) - Number(a.expected_loss_if_approved || 0)
    );
    setItems(sorted);
  }

  useEffect(() => {
    load().catch(console.error);
    const t = setInterval(() => load().catch(() => {}), 3000);
    return () => clearInterval(t);
  }, []);

  async function feedback(id, label) {
    await api.feedback({ transaction_id: id, analyst_label: label });
    setMsg(`Updated ${id}`);
    load();
  }

  return (
    <div className="space-y-5 animate-rise">
      <div>
        <h1 className="font-display text-4xl font-bold">Review Queue</h1>
        <p className="mt-2 text-slate-400">Uncertain or borderline cases sorted by expected loss if approved.</p>
      </div>
      {msg && <div className="text-sm text-mint-400">{msg}</div>}
      <div className="panel divide-y divide-white/5">
        {items.map((r) => (
          <div key={r.transaction_id} className="flex flex-wrap items-center gap-3 px-4 py-3">
            <DecisionBadge decision={r.decision} />
            <Link to={`/tx/${r.transaction_id}`} className="font-mono text-sm text-mint-400 hover:underline">
              {r.transaction_id}
            </Link>
            <span className="text-sm text-slate-400">p={Number(r.p_fraud).toFixed(3)}</span>
            <span className="text-sm text-slate-400">σ={Number(r.uncertainty_std).toFixed(3)}</span>
            <span className="ml-auto font-mono text-sm">E[loss] {Number(r.expected_loss_if_approved).toFixed(2)}</span>
            <button className="btn-primary !py-1.5" type="button" onClick={() => feedback(r.transaction_id, 1)}>
              Fraud
            </button>
            <button className="btn-ghost !py-1.5" type="button" onClick={() => feedback(r.transaction_id, 0)}>
              Genuine
            </button>
          </div>
        ))}
        {!items.length && <div className="px-4 py-12 text-center text-slate-500">No review cases yet — start the live stream.</div>}
      </div>
    </div>
  );
}
