const STYLES = {
  APPROVE: "bg-mint-500/15 text-mint-400 ring-1 ring-mint-500/30",
  REVIEW: "bg-warn-500/15 text-warn-400 ring-1 ring-warn-500/30",
  BLOCK: "bg-alert-500/15 text-alert-400 ring-1 ring-alert-500/30",
};

export default function DecisionBadge({ decision }) {
  return <span className={`badge ${STYLES[decision] || "bg-slate-500/20 text-slate-300"}`}>{decision}</span>;
}
