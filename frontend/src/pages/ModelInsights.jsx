import { useEffect, useState } from "react";
import { api } from "../api";

export default function ModelInsights() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    api.metrics().then(setMetrics).catch(console.error);
  }, []);

  const mm = metrics?.model_metrics || {};
  const test = mm.test || {};
  const lr = mm.lr_test || {};

  const rows = [
    ["Random Forest (calibrated)", test],
    ["Logistic Regression", lr],
  ];

  return (
    <div className="space-y-5 animate-rise">
      <div>
        <h1 className="font-display text-4xl font-bold">Model Insights</h1>
        <p className="mt-2 text-slate-400">Time-based test metrics — PR-AUC is primary (accuracy is misleading under 0.17% fraud).</p>
      </div>
      <div className="panel overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="border-b border-white/10 text-xs uppercase text-slate-500">
            <tr>
              {["Model", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "F2", "MCC"].map((h) => (
                <th key={h} className="px-4 py-3 text-left">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, m]) => (
              <tr key={name} className="border-b border-white/5">
                <td className="px-4 py-3 font-medium">{name}</td>
                {["pr_auc", "roc_auc", "precision", "recall", "f1", "f2", "mcc"].map((k) => (
                  <td key={k} className="px-4 py-3 font-mono">
                    {m[k] != null ? Number(m[k]).toFixed(4) : "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <div className="panel p-4">
          <div className="text-xs uppercase text-slate-500">Test ECE</div>
          <div className="font-display text-3xl">{mm.test_ece != null ? Number(mm.test_ece).toFixed(4) : "—"}</div>
        </div>
        <div className="panel p-4">
          <div className="text-xs uppercase text-slate-500">Recall @ 90% precision</div>
          <div className="font-display text-3xl">
            {mm.test_recall_at_90_precision != null ? Number(mm.test_recall_at_90_precision).toFixed(4) : "—"}
          </div>
        </div>
        <div className="panel p-4">
          <div className="text-xs uppercase text-slate-500">Live cost saved (est.)</div>
          <div className="font-display text-3xl">{metrics ? Number(metrics.cost_saved_est).toFixed(1) : "—"}</div>
        </div>
      </div>
      <p className="text-sm text-slate-500">
        Ablation / ROC-PR figures from full training pipeline land in <code className="font-mono">reports/figures/</code>. This page reads live API metrics from the web scoring bundle.
      </p>
    </div>
  );
}
