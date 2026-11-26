import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, liveSocket } from "../api";
import DecisionBadge from "../components/DecisionBadge";

function Kpi({ label, value, hint }) {
  return (
    <div className="panel animate-rise p-4">
      <div className="text-xs uppercase tracking-wider text-slate-400">{label}</div>
      <div className="mt-1 font-display text-3xl font-semibold text-white">{value}</div>
      {hint && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

export default function LiveMonitor() {
  const [rows, setRows] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [rate, setRate] = useState(5);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api.metrics().then(setMetrics).catch((e) => setError(String(e.message || e)));
    api.transactions({ limit: 40 }).then((d) => setRows(d.items || [])).catch(() => {});
    const ws = liveSocket((msg) => {
      setRows((prev) => [msg, ...prev].slice(0, 80));
    });
    const t = setInterval(() => {
      api.metrics().then((m) => {
        setMetrics(m);
        setRunning(!!m.simulator_running);
      }).catch(() => {});
    }, 2000);
    return () => {
      ws.close();
      clearInterval(t);
    };
  }, []);

  const chartData = useMemo(
    () =>
      [...rows]
        .slice(0, 40)
        .reverse()
        .map((r, i) => ({ i, risk: Number(r.risk_score || 0), p: Number(r.p_fraud || 0) })),
    [rows]
  );

  async function start() {
    setError("");
    try {
      await api.streamStart({ rate_tps: rate, dataset_slice: "test" });
      setRunning(true);
    } catch (e) {
      setError(String(e.message || e));
    }
  }
  async function stop() {
    await api.streamStop();
    setRunning(false);
  }

  return (
    <div className="space-y-6">
      <section className="animate-rise">
        <h1 className="font-display text-4xl font-bold text-white md:text-5xl">Live Monitor</h1>
        <p className="mt-2 max-w-2xl text-slate-400">
          Stream scored payments with Approve / Review / Block decisions, uncertainty, and latency — under 100&nbsp;ms target on CPU.
        </p>
      </section>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi label="Scored" value={metrics?.volume ?? "—"} hint="session volume" />
        <Kpi label="Block rate" value={metrics ? `${(metrics.block_rate * 100).toFixed(1)}%` : "—"} />
        <Kpi label="Avg latency" value={metrics ? `${metrics.avg_latency_ms.toFixed(1)} ms` : "—"} />
        <Kpi label="Blocked amount" value={metrics ? metrics.amount_blocked.toFixed(0) : "—"} hint="currency units" />
      </div>

      <div className="panel flex flex-wrap items-end gap-3 p-4">
        <label className="text-sm text-slate-300">
          Speed (tx/s)
          <input
            className="input mt-1 w-28"
            type="number"
            min={1}
            max={100}
            value={rate}
            onChange={(e) => setRate(Number(e.target.value))}
          />
        </label>
        {!running ? (
          <button className="btn-primary" type="button" onClick={start}>
            Start simulator
          </button>
        ) : (
          <button className="btn-ghost text-alert-400" type="button" onClick={stop}>
            Stop
          </button>
        )}
        <span className="font-mono text-xs text-slate-500">{running ? `RUNNING @ ${rate} tps` : "IDLE"}</span>
        {error && <span className="text-sm text-alert-400">{error}</span>}
      </div>

      <div className="panel h-56 p-4">
        <div className="mb-2 text-sm text-slate-400">Live risk score</div>
        <ResponsiveContainer width="100%" height="90%">
          <LineChart data={chartData}>
            <XAxis dataKey="i" hide />
            <YAxis domain={[0, 1]} stroke="#64748b" width={32} />
            <Tooltip contentStyle={{ background: "#111827", border: "1px solid #334155" }} />
            <Line type="monotone" dataKey="risk" stroke="#10b981" dot={false} strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="panel overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="border-b border-white/10 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-4 py-3">ID</th>
                <th className="px-4 py-3">Amount</th>
                <th className="px-4 py-3">p(fraud)</th>
                <th className="px-4 py-3">Uncertainty</th>
                <th className="px-4 py-3">Decision</th>
                <th className="px-4 py-3">Latency</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.transaction_id + String(r.latency_ms)} className="border-b border-white/5 hover:bg-white/[0.03]">
                  <td className="px-4 py-2.5 font-mono text-xs">
                    <Link className="text-mint-400 hover:underline" to={`/tx/${r.transaction_id}`}>
                      {r.transaction_id}
                    </Link>
                  </td>
                  <td className="px-4 py-2.5">{Number(r.amount ?? 0).toFixed(2)}</td>
                  <td className="px-4 py-2.5 font-mono">{Number(r.p_fraud).toFixed(3)}</td>
                  <td className="px-4 py-2.5 font-mono">{Number(r.uncertainty_std).toFixed(3)}</td>
                  <td className="px-4 py-2.5">
                    <DecisionBadge decision={r.decision} />
                  </td>
                  <td className="px-4 py-2.5 font-mono text-slate-400">{Number(r.latency_ms).toFixed(1)} ms</td>
                </tr>
              ))}
              {!rows.length && (
                <tr>
                  <td colSpan={6} className="px-4 py-10 text-center text-slate-500">
                    Start the simulator to see live decisions.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
