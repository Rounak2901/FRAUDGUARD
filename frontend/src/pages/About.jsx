export default function About() {
  return (
    <div className="prose-invert max-w-3xl animate-rise space-y-4">
      <h1 className="font-display text-4xl font-bold text-white">About Fraud Guard</h1>
      <p className="text-slate-300 leading-relaxed">
        Online payment fraud is rare (often &lt;1%), fast, and expensive. Accuracy looks great if you always predict
        “genuine” — and catches nothing. This system scores each payment with a calibrated model, estimates
        uncertainty (tree-ensemble epistemic stand-in for a Bayesian NN), maintains Beta beliefs per entity, adds an
        anomaly score, and chooses Approve / Review / Block to minimise expected business cost.
      </p>
      <h2 className="font-display text-2xl text-white">Pipeline</h2>
      <ol className="list-decimal space-y-2 pl-5 text-slate-300">
        <li>Validate transaction → build features</li>
        <li>Bayesian entity-risk lookup (Beta posterior)</li>
        <li>Supervised probability + uncertainty</li>
        <li>Isolation Forest anomaly score</li>
        <li>Cost-sensitive decision engine</li>
        <li>Local explanation reasons for analysts</li>
      </ol>
      <h2 className="font-display text-2xl text-white">Limitations</h2>
      <p className="text-slate-400">
        Public ULB features are PCA-anonymised; entity IDs may be synthetic; the stream is a replay simulator, not a
        live bank feed. Results are for research and demonstration — not production compliance (PCI-DSS).
      </p>
    </div>
  );
}
