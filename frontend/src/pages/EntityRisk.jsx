import { useState } from "react";
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";

export default function EntityRisk() {
  const [id, setId] = useState("card_0");
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");

  async function search(e) {
    e?.preventDefault();
    setErr("");
    try {
      setData(await api.entity(id.trim()));
    } catch (ex) {
      setErr(String(ex.message || ex));
    }
  }

  const hist = (data?.history || []).map((h, i) => ({ i, mean: h.mean, alpha: h.alpha, beta: h.beta }));

  return (
    <div className="space-y-5 animate-rise">
      <div>
        <h1 className="font-display text-4xl font-bold">Entity Risk Explorer</h1>
        <p className="mt-2 text-slate-400">Beta(α, β) fraud belief per card / user — updates when analysts confirm outcomes.</p>
      </div>
      <form className="panel flex flex-wrap gap-3 p-4" onSubmit={search}>
        <input className="input max-w-sm" value={id} onChange={(e) => setId(e.target.value)} placeholder="card_id / entity_id" />
        <button className="btn-primary" type="submit">Lookup</button>
      </form>
      {err && <div className="text-alert-400">{err}</div>}
      {data && (
        <div className="grid gap-4 md:grid-cols-2">
          <div className="panel p-5">
            <div className="text-xs uppercase text-slate-500">{data.entity_id}</div>
            <div className="mt-2 font-display text-4xl text-mint-400">{(data.mean * 100).toFixed(3)}%</div>
            <div className="mt-3 space-y-1 font-mono text-sm text-slate-300">
              <div>α = {data.alpha.toFixed(3)}</div>
              <div>β = {data.beta.toFixed(3)}</div>
              <div>
                95% CI [{data.credible_interval_95[0].toFixed(4)}, {data.credible_interval_95[1].toFixed(4)}]
              </div>
            </div>
          </div>
          <div className="panel h-64 p-4">
            <div className="mb-2 text-sm text-slate-400">Posterior mean over feedback updates</div>
            <ResponsiveContainer width="100%" height="90%">
              <AreaChart data={hist.length ? hist : [{ i: 0, mean: data.mean }]}>
                <XAxis dataKey="i" hide />
                <YAxis domain={[0, "auto"]} stroke="#64748b" width={40} />
                <Tooltip contentStyle={{ background: "#111827", border: "1px solid #334155" }} />
                <Area type="monotone" dataKey="mean" stroke="#10b981" fill="rgba(16,185,129,0.2)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
