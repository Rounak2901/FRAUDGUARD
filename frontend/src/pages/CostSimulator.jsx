import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";

export default function CostSimulator() {
  const [cFp, setCFp] = useState(5);
  const [cFn, setCFn] = useState(1);
  const [cReview, setCReview] = useState(2);
  const [result, setResult] = useState(null);

  useEffect(() => {
    const t = setTimeout(() => {
      api
        .costSimulate({ c_fp: cFp, c_fn_multiplier: cFn, c_review: cReview })
        .then(setResult)
        .catch(console.error);
    }, 200);
    return () => clearTimeout(t);
  }, [cFp, cFn, cReview]);

  return (
    <div className="space-y-5 animate-rise">
      <div>
        <h1 className="font-display text-4xl font-bold">Cost Simulator</h1>
        <p className="mt-2 text-slate-400">Tune false-positive friction vs missed-fraud loss — optimal threshold updates live.</p>
      </div>
      <div className="panel grid gap-4 p-5 md:grid-cols-3">
        <label className="text-sm">
          C_fp (investigation friction)
          <input className="input mt-1" type="range" min={1} max={50} value={cFp} onChange={(e) => setCFp(Number(e.target.value))} />
          <div className="font-mono text-mint-400">{cFp}</div>
        </label>
        <label className="text-sm">
          C_fn multiplier × amount
          <input className="input mt-1" type="range" min={0.2} max={5} step={0.1} value={cFn} onChange={(e) => setCFn(Number(e.target.value))} />
          <div className="font-mono text-mint-400">{cFn.toFixed(1)}</div>
        </label>
        <label className="text-sm">
          C_review
          <input className="input mt-1" type="range" min={0.5} max={20} step={0.5} value={cReview} onChange={(e) => setCReview(Number(e.target.value))} />
          <div className="font-mono text-mint-400">{cReview}</div>
        </label>
      </div>
      {result && (
        <>
          <div className="grid gap-3 sm:grid-cols-4">
            {["APPROVE", "REVIEW", "BLOCK"].map((k) => (
              <div key={k} className="panel p-4">
                <div className="text-xs text-slate-500">{k}</div>
                <div className="font-display text-2xl">{result.totals[k]}</div>
              </div>
            ))}
            <div className="panel p-4">
              <div className="text-xs text-slate-500">Optimal threshold</div>
              <div className="font-display text-2xl">{result.optimal_threshold.threshold.toFixed(2)}</div>
              <div className="text-xs text-slate-500">cost {result.optimal_threshold.total_cost.toFixed(1)}</div>
            </div>
          </div>
          <div className="panel h-72 p-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={result.cost_curve}>
                <CartesianGrid stroke="rgba(148,163,184,0.1)" />
                <XAxis dataKey="threshold" stroke="#64748b" />
                <YAxis stroke="#64748b" />
                <Tooltip contentStyle={{ background: "#111827", border: "1px solid #334155" }} />
                <Line type="monotone" dataKey="total_cost" stroke="#38bdf8" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
