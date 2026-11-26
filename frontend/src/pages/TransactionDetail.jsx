import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import DecisionBadge from "../components/DecisionBadge";

export default function TransactionDetail() {
  const { id } = useParams();
  const [tx, setTx] = useState(null);
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");

  useEffect(() => {
    api.transaction(id).then(setTx).catch((e) => setErr(String(e.message || e)));
  }, [id]);

  async function feedback(label) {
    setMsg("");
    try {
      const res = await api.feedback({ transaction_id: id, analyst_label: label });
      setMsg(`Feedback saved. Entity ${res.entity_id} posterior mean → ${res.belief.mean.toFixed(4)}`);
    } catch (e) {
      setErr(String(e.message || e));
    }
  }

  if (err && !tx) return <div className="text-alert-400">{err}</div>;
  if (!tx) return <div className="text-slate-400">Loading…</div>;

  const shap = (tx.top_reasons || []).map((r) => ({
    name: r.feature,
    impact: r.impact,
    fill: r.impact >= 0 ? "#f43f5e" : "#10b981",
  }));

  return (
    <div className="space-y-6 animate-rise">
      <div className="flex flex-wrap items-center gap-3">
        <Link to="/" className="text-sm text-slate-400 hover:text-mint-400">
          ← Live
        </Link>
        <h1 className="font-display text-3xl font-bold">{tx.transaction_id}</h1>
        <DecisionBadge decision={tx.decision} />
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="panel p-5 md:col-span-1">
          <div className="text-xs uppercase text-slate-500">Fraud probability</div>
          <div className="mt-2 font-display text-5xl font-bold text-mint-400">{(tx.p_fraud * 100).toFixed(1)}%</div>
          <div className="mt-2 font-mono text-xs text-slate-400">
            95% CI [{tx.credible_interval_95?.[0]?.toFixed(3)}, {tx.credible_interval_95?.[1]?.toFixed(3)}]
          </div>
          <div className="mt-4 space-y-2 text-sm">
            <div className="flex justify-between"><span className="text-slate-400">Uncertainty σ</span><span className="font-mono">{tx.uncertainty_std?.toFixed(4)}</span></div>
            <div className="flex justify-between"><span className="text-slate-400">Anomaly</span><span className="font-mono">{tx.anomaly_score?.toFixed(4)}</span></div>
            <div className="flex justify-between"><span className="text-slate-400">Entity risk</span><span className="font-mono">{tx.entity_risk_posterior?.toFixed(4)}</span></div>
            <div className="flex justify-between"><span className="text-slate-400">Risk score</span><span className="font-mono">{tx.risk_score?.toFixed(4)}</span></div>
            <div className="flex justify-between"><span className="text-slate-400">E[loss] if approved</span><span className="font-mono">{tx.expected_loss_if_approved?.toFixed(2)}</span></div>
            <div className="flex justify-between"><span className="text-slate-400">Latency</span><span className="font-mono">{tx.latency_ms?.toFixed(1)} ms</span></div>
          </div>
          <div className="mt-5 flex gap-2">
            <button className="btn-primary flex-1" type="button" onClick={() => feedback(1)}>Confirm Fraud</button>
            <button className="btn-ghost flex-1" type="button" onClick={() => feedback(0)}>Confirm Genuine</button>
          </div>
          {msg && <p className="mt-3 text-sm text-mint-400">{msg}</p>}
        </div>

        <div className="panel p-5 md:col-span-2">
          <h2 className="font-display text-xl">SHAP-style local impacts</h2>
          <div className="mt-4 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={shap} layout="vertical" margin={{ left: 40 }}>
                <XAxis type="number" stroke="#64748b" />
                <YAxis type="category" dataKey="name" width={90} stroke="#94a3b8" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ background: "#111827", border: "1px solid #334155" }} />
                <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                  {shap.map((e, i) => (
                    <Cell key={i} fill={e.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <ul className="mt-4 space-y-2">
            {(tx.top_reasons || []).map((r) => (
              <li key={r.feature} className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2 text-sm text-slate-300">
                {r.text}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
