"""Load reports/results.json and print a concise evaluation summary."""

from __future__ import annotations

import json

from src.config import load_config, ROOT


def main() -> None:
    cfg = load_config()
    path = ROOT / cfg["paths"]["results_file"]
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: py -m src.train")
    results = json.loads(path.read_text(encoding="utf-8"))
    print("=== Fraud Guard evaluation summary ===")
    if "baselines" in results:
        print("\nBaselines (test PR-AUC):")
        for k, v in results["baselines"].items():
            print(f"  {k:22s} {v.get('pr_auc', 0):.4f}")
    if "nn" in results:
        print(f"\nNN ({results['nn'].get('chosen_loss')}): {results['nn']['test']['pr_auc']:.4f}")
    if "bnn" in results:
        print(f"BNN calibrated:          {results['bnn']['test_cal']['pr_auc']:.4f}")
        print(f"ECE before→after:        {results['bnn']['ece_before']:.4f} → {results['bnn']['ece_after']:.4f}")
    if "ensemble" in results:
        print(f"Ensemble:                {results['ensemble']['test']['pr_auc']:.4f}")
    if "ablation" in results:
        print("\nAblation (test PR-AUC / cost@0.5):")
        for k, v in results["ablation"].items():
            print(f"  {k:22s} PR={v['pr_auc']:.4f}  cost={v['business_cost_at_0_5']:.1f}")
    if "decision" in results:
        d = results["decision"]
        print(f"\nCost @ val-opt thr (test): {d.get('test_cost_at_val_threshold')}")
        print(f"Cost @ 0.5 thr (test):     {d.get('test_cost_at_0_5')}")


if __name__ == "__main__":
    main()
