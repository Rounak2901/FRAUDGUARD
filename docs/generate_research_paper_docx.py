"""Generate Word (.docx) research paper. Run: py docs/generate_research_paper_docx.py"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT_DOCS = ROOT / "docs" / "FraudGuard_Research_Paper.docx"
OUT_ROOT = ROOT / "FraudGuard_Research_Paper.docx"


def set_run_font(run, size=11, bold=False, italic=False, name="Times New Roman"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def add_para(doc, text, *, size=11, bold=False, italic=False, align="justify", space_after=8):
    p = doc.add_paragraph()
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "left":
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def add_heading_custom(doc, text, level=1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    set_run_font(run, size=13 if level == 1 else 12, bold=True)
    run.font.color.rgb = RGBColor(0x0F, 0x27, 0x44)
    return p


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        run = p.add_run(h)
        set_run_font(run, size=9, bold=True)
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            cell = table.rows[r_i + 1].cells[c_i]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            set_run_font(run, size=9)
    doc.add_paragraph()


def build():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)

    add_para(
        doc,
        "Cost-Sensitive Bayesian Neural Detection of Online Payment Fraud "
        "with Entity-Level Risk Updating and Real-Time Explainable Decisions",
        size=16,
        bold=True,
        align="center",
        space_after=10,
    )
    add_para(doc, "Rounak Dhall", size=12, align="center", space_after=4)
    add_para(
        doc,
        "Department of Computer Science and Engineering (B.Tech.), Independent Research Prototype — Fraud Guard System",
        size=10,
        italic=True,
        align="center",
        space_after=2,
    )
    add_para(
        doc,
        "Corresponding author: rounak2901@gmail.com | Code: https://github.com/Rounak2901/FRAUDGUARD",
        size=10,
        italic=True,
        align="center",
        space_after=14,
    )

    add_heading_custom(doc, "Abstract")
    add_para(
        doc,
        "Online payment fraud is rare, fast, and asymmetric in cost: missing a fraudulent transfer "
        "typically destroys far more value than incorrectly challenging a legitimate customer. "
        "Accuracy is therefore a misleading headline metric under extreme class imbalance. "
        "This paper presents Fraud Guard, an end-to-end detection architecture that couples "
        "(i) a feed-forward neural classifier, (ii) Monte Carlo Dropout to quantify epistemic uncertainty, "
        "(iii) conjugate Beta–Bernoulli updating of per-entity fraud risk for cards and merchants, "
        "(iv) an unsupervised anomaly score trained only on genuine behaviour, and "
        "(v) a cost-sensitive decision engine that maps scores to Approve, Review, or Block actions. "
        "Local explanations are produced for analyst review, and the full stack is exposed through a "
        "low-latency scoring API and live dashboard. Evaluation uses a time-ordered train/validation/test "
        "protocol to avoid temporal leakage. On a reproducible laptop-scale benchmark with a public-style "
        "feature schema, the neural model attained a test precision–recall area under the curve (PR-AUC) "
        "of 0.9265, isotonic calibration reduced expected calibration error from 0.1803 to 0.0005, and "
        "threshold optimisation lowered estimated business cost relative to a default 0.5 cut-off "
        "(1013.2 versus 5568.4 in the reported currency units). The study emphasises honest reporting: "
        "results depend on the available public or synthetic streams and should be re-validated on "
        "institution-specific ledgers before operational deployment.",
        size=10,
    )
    add_para(
        doc,
        "Keywords: payment fraud detection; class imbalance; PR-AUC; Bayesian neural networks; "
        "Monte Carlo Dropout; Beta–Bernoulli updating; cost-sensitive learning; anomaly detection; "
        "explainable AI; real-time scoring",
        size=10,
        italic=True,
        space_after=12,
    )

    add_heading_custom(doc, "1. Introduction")
    add_para(
        doc,
        "Digital commerce concentrates payment risk into millisecond decisions. Issuers and processors "
        "must score each authorisation request before funds move, while fraudsters continuously adapt "
        "to static rules. Contemporary detection pipelines therefore combine supervised learning on "
        "labelled histories with streaming features, risk scores, and human review queues. Yet three "
        "gaps recur in practice. First, many published studies still emphasise accuracy or ROC-AUC even "
        "when the fraud prior is below one percent, which can flatter models that rarely raise alarms. "
        "Second, point estimates of fraud probability are treated as certain, so borderline cases are "
        "auto-decided when they should be escalated. Third, entity memory—how risky a card or merchant "
        "has been recently—is often encoded only as ad-hoc aggregates rather than as an explicit "
        "probabilistic belief that updates when analysts confirm outcomes.",
    )
    add_para(
        doc,
        "This work addresses those gaps with a unified prototype named Fraud Guard. The design goal is "
        "not merely to maximise a ranking metric, but to minimise expected operational loss under a "
        "configurable cost matrix while remaining interpretable to analysts and deployable on commodity "
        "CPU hardware. The contributions are:",
    )
    for item in [
        "A time-respecting training protocol with robust amount scaling fit on the training window only, and causal entity features that never peek into the future.",
        "A Bayesian neural layer via Monte Carlo Dropout that yields predictive means, uncertainty, and approximate credible intervals used to open a Review band.",
        "Online Beta–Bernoulli entity-risk beliefs with mild temporal decay, updated from analyst feedback and injected into both the ensemble score and the dashboard.",
        "A cost-sensitive Approve/Review/Block engine with validation-set threshold search, ablation of ensemble components, and a working real-time API with explainable reasons.",
    ]:
        p = doc.add_paragraph(style="List Number")
        p.clear()
        run = p.add_run(item)
        set_run_font(run, size=11)
        p.paragraph_format.space_after = Pt(4)

    add_heading_custom(doc, "2. Related Work")
    add_para(
        doc,
        "Supervised fraud classifiers on anonymised credit-card ledgers are well studied. Classical "
        "baselines such as logistic regression, random forests, and gradient boosting remain strong "
        "references under careful validation. Deep architectures—including multilayer perceptrons, "
        "recurrent networks, and sequence models—have been explored for capturing temporal dependence, "
        "sometimes with Bayesian hyperparameter search to stabilise training under imbalance. Parallel "
        "streams investigate unsupervised or hybrid detectors when labels are delayed, for example "
        "isolation forests and autoencoders trained on genuine traffic.",
    )
    add_para(
        doc,
        "Survey and critical analyses of financial fraud analytics emphasise that discrete rule checks "
        "on accounts or devices are insufficient against adaptive adversaries, and that supervised "
        "anomaly methods must be complemented by unsupervised cues. Recent Bayesian ensemble proposals "
        "combine multiple learners and reliability fusion to improve robustness on synthetic mobile-money "
        "streams. Our work differs in focus: rather than introducing another opaque high-capacity "
        "ensemble, we organise uncertainty, entity beliefs, anomaly evidence, and dollar costs into an "
        "operational decision policy with explicit review routing and live explainability.",
    )

    add_heading_custom(doc, "3. Problem Formulation")
    add_para(
        doc,
        "Let each transaction be a feature vector x with amount a and binary label y ∈ {0,1} "
        "(1 = fraud). A scoring function returns a probability p̂ = P(y=1|x) and an uncertainty summary "
        "u(x). Actions are drawn from {Approve, Review, Block}. We adopt a simple but transparent "
        "cost model: a false positive incurs a fixed friction cost C_fp (investigation or customer "
        "inconvenience), while a false negative loses approximately a · C_fn. Review incurs C_review "
        "and is preferred when u(x) is large or when p̂ lies near the cost-optimal boundary. "
        "The primary ranking metric is PR-AUC on a chronologically held-out test window; secondary "
        "metrics include ROC-AUC, F2, Matthews correlation (MCC), Brier score, and expected "
        "calibration error (ECE).",
    )

    add_heading_custom(doc, "4. Proposed System: Fraud Guard")
    add_heading_custom(doc, "4.1 Data pipeline and leakage controls", level=2)
    add_para(
        doc,
        "Transactions are sorted by time and partitioned into contiguous 70%/15%/15% blocks for "
        "training, validation, and testing. Amount is log-transformed and robust-scaled with "
        "statistics estimated on the training block alone. Hour-of-day is encoded with sine/cosine "
        "harmonics. For streaming entity memory we maintain per-card rolling counts, amount moments, "
        "velocity, and time since last purchase using only past events for that entity.",
    )
    add_heading_custom(doc, "4.2 Supervised neural classifier", level=2)
    add_para(
        doc,
        "The core model is a three-hidden-layer multilayer perceptron (128–64–32 units) with batch "
        "normalisation, ReLU activations, and dropout. Training compares class-weighted binary "
        "cross-entropy against focal loss; early stopping monitors validation PR-AUC. Fixed seeds "
        "support reproducibility on CPU-only laptops.",
    )
    add_heading_custom(doc, "4.3 Uncertainty via Monte Carlo Dropout", level=2)
    add_para(
        doc,
        "Following the interpretation of dropout as approximate variational inference, we keep dropout "
        "active at test time and draw T stochastic forward passes (T = 40 in experiments). The mean "
        "probability estimates fraud likelihood; the empirical standard deviation proxies epistemic "
        "uncertainty. A normal approximation yields a 95% interval used in the user interface. "
        "Isotonic regression fitted on the validation window recalibrates probabilities before "
        "decisioning.",
    )
    add_heading_custom(doc, "4.4 Bayesian entity-risk updating", level=2)
    add_para(
        doc,
        "For each entity identifier we maintain a Beta(α, β) belief with a weakly informative prior "
        "matched to a low global fraud rate (e.g., α0=1, β0=500). Confirmed fraud increments α; "
        "confirmed genuine increments β. A daily multiplicative decay gently forgets stale evidence. "
        "The posterior mean enters the ensemble and is visualised live after analyst feedback.",
    )
    add_heading_custom(doc, "4.5 Anomaly channel and ensemble", level=2)
    add_para(
        doc,
        "An autoencoder trained exclusively on genuine training rows supplies a reconstruction-error "
        "anomaly score, min–max normalised using validation statistics. The final risk score is a "
        "convex combination of calibrated neural probability, entity posterior mean, and anomaly score. "
        "Mixture weights are selected by grid search on validation PR-AUC.",
    )
    add_heading_custom(doc, "4.6 Cost-sensitive decisions and explanations", level=2)
    add_para(
        doc,
        "On the validation window we sweep decision thresholds to minimise total monetary cost under "
        "the configured matrix, then freeze the threshold for the test window. High uncertainty or "
        "proximity to the boundary triggers Review. Local reasons are produced from a gradient-boosting "
        "surrogate TreeExplainer when available, otherwise from signed standardised deviations weighted "
        "by linear coefficients, and rendered as plain-language bullets for analysts.",
    )
    add_heading_custom(doc, "4.7 Serving prototype", level=2)
    add_para(
        doc,
        "A FastAPI service validates inputs, scores transactions, persists decisions, updates beliefs "
        "from feedback, and streams events over WebSocket. A React dashboard provides a live monitor, "
        "review queue, entity explorer, cost simulator, and manual “try-it” form for demonstrations.",
    )

    add_heading_custom(doc, "5. Experimental Setup")
    add_para(
        doc,
        "Experiments were executed on a standard CPU laptop using Python, PyTorch, scikit-learn, and "
        "XGBoost. The feature schema follows the widely used anonymised credit-card fraud setting "
        "(time, amount, and PCA components). Because institutional ledgers cannot be redistributed, "
        "the quantitative tables in this manuscript correspond to a fully reproducible synthetic stream "
        "that matches the public schema and a similar fraud prevalence (~0.17%). The same code path "
        "accepts the original Kaggle credit-card file and a PaySim sample for entity trajectories when "
        "credentials are available. Classical baselines (logistic regression, random forest, XGBoost, "
        "isolation forest) share the identical chronological split and metric suite.",
    )

    add_heading_custom(doc, "6. Results and Discussion")
    add_heading_custom(doc, "6.1 Ranking performance", level=2)
    add_para(
        doc,
        "Table 1 summarises test-set ranking quality. Logistic regression remained a formidable baseline "
        "(PR-AUC 0.9324). The multilayer perceptron with weighted BCE reached 0.9265 PR-AUC and near-ceiling "
        "ROC-AUC, confirming that a compact network is competitive when leakage is controlled. XGBoost "
        "achieved 0.8026 PR-AUC with balanced precision and recall at the default 0.5 threshold. "
        "Isolation forest ranked poorly on PR-AUC, as expected for a purely unsupervised score under "
        "severe imbalance, yet remained useful as a complementary anomaly channel.",
    )
    add_table(
        doc,
        ["Model", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "MCC"],
        [
            ["Logistic Regression", "0.9324", "0.9996", "0.4595", "0.9444", "0.6182", "0.6577"],
            ["Random Forest", "0.6303", "0.9960", "1.0000", "0.1667", "0.2857", "0.4078"],
            ["XGBoost", "0.8026", "0.9981", "0.7222", "0.7222", "0.7222", "0.7216"],
            ["Isolation Forest", "0.0565", "0.9408", "0.0139", "0.8333", "0.0273", "0.0963"],
            ["MLP (weighted BCE)", "0.9265", "0.9996", "0.0456", "1.0000", "0.0872", "0.2080"],
            ["MC-Dropout + isotonic", "0.8216", "0.9443", "0.9286", "0.7222", "0.8125", "0.8186"],
            ["Full ensemble score", "0.8908", "0.9918", "—*", "—*", "—*", "—*"],
        ],
    )
    add_para(
        doc,
        "Table 1. Test metrics at threshold 0.5 unless noted. *Ensemble precision/recall at 0.5 are "
        "unstable because the learned mixture shifts mass; operational performance is reported after "
        "cost-based thresholding (Section 6.3).",
        size=9,
        italic=True,
        align="center",
    )

    add_heading_custom(doc, "6.2 Calibration and uncertainty quality", level=2)
    add_para(
        doc,
        "Raw MC-Dropout means were over-confident on this stream (ECE 0.1803). After isotonic "
        "calibration on validation probabilities, ECE fell to 0.0005. Average uncertainty was higher "
        "on erroneous predictions (0.139) than on correct ones (0.085), supporting the use of "
        "uncertainty to gate Review rather than forcing binary auto-decisions.",
    )

    add_heading_custom(doc, "6.3 Cost optimisation and ablation", level=2)
    add_para(
        doc,
        "Optimising the decision threshold on validation cost yielded a substantially lower test loss "
        "than the conventional 0.5 cut-off (1013.2 versus 5568.4 under the configured matrix). "
        "Ablation showed that adding the anomaly channel improved PR-AUC relative to the calibrated "
        "network alone (0.8908 versus 0.8216). Entity beliefs primarily benefit live adaptation after "
        "feedback; their offline contribution depends on how richly identifiers repeat in the sample.",
    )
    add_table(
        doc,
        ["Configuration", "Test PR-AUC / Cost", "Notes"],
        [
            ["Calibrated NN only", "0.8216", "MC-Dropout + isotonic"],
            ["NN + entity posterior", "0.8040", "Soft Beta mean features"],
            ["NN + anomaly", "0.8908", "Autoencoder residual"],
            ["Full ensemble", "0.8908", "Grid weights on validation"],
            ["Cost @ thr=0.5", "5568.4*", "Higher operational loss"],
            ["Cost @ val-opt thr", "1013.2*", "Selected then frozen"],
        ],
    )
    add_para(
        doc,
        "Table 2. Ablation and cost comparison (*cost in experiment currency units).",
        size=9,
        italic=True,
        align="center",
    )

    add_heading_custom(doc, "6.4 Discussion", level=2)
    add_para(
        doc,
        "Two practical lessons emerge. First, PR-AUC and cost can disagree with thresholded F1: the "
        "MLP’s default 0.5 operating point was overly aggressive (high recall, low precision), yet "
        "ranking quality remained strong—reinforcing the need for cost-aware thresholding. Second, "
        "calibration and uncertainty are not optional decorations; they determine whether a model’s "
        "probability can safely drive money-moving actions or only a review queue. The serving "
        "prototype demonstrates that these ideas can coexist with interactive explanations without "
        "requiring a GPU cluster.",
    )

    add_heading_custom(doc, "7. Limitations and Threats to Validity")
    add_para(
        doc,
        "Public PCA features limit semantic interpretability of attributions. Synthetic entity "
        "identifiers, when used, do not reproduce real multi-hop fraud rings. The reported numbers "
        "are tied to a reproducible synthetic draw unless replaced with licensed Kaggle extracts; "
        "external validity to any single bank remains unproven. The adversary is passive in our "
        "evaluation—no adaptive evasion is simulated. Latency measurements exclude wide-area network "
        "variance of production payment switches.",
    )

    add_heading_custom(doc, "8. Conclusion and Future Work")
    add_para(
        doc,
        "Fraud Guard integrates calibrated Bayesian neural scoring, entity-level Beta risk memory, "
        "anomaly evidence, and cost-sensitive Approve/Review/Block policies into a transparent "
        "real-time prototype. Empirically, chronological evaluation, calibration, and threshold "
        "optimisation matter as much as raw classifier choice. Future work includes graph neural "
        "models over transaction networks, online drift detectors, adversarial robustness tests, "
        "and deployment-oriented MLOps with continuous feedback from confirmed cases.",
    )

    add_heading_custom(doc, "Acknowledgments")
    add_para(
        doc,
        "The author thanks open dataset curators whose public benchmarks enable comparable fraud "
        "research, and notes that this manuscript describes an academic prototype rather than a "
        "certified payment product.",
    )

    add_heading_custom(doc, "References")
    refs = [
        "[1] A. Dal Pozzolo, O. Caelen, R. A. Johnson, and G. Bontempi, “Calibrating probability with undersampling for unbalanced classification,” in Proc. IEEE Symp. Computational Intelligence and Data Mining, 2015.",
        "[2] Y. Gal and Z. Ghahramani, “Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,” in Proc. ICML, 2016.",
        "[3] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal loss for dense object detection,” in Proc. IEEE ICCV, 2017.",
        "[4] F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation forest,” in Proc. IEEE ICDM, 2008.",
        "[5] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in Proc. NeurIPS, 2017.",
        "[6] C. Elkan, “The foundations of cost-sensitive learning,” in Proc. IJCAI, 2001.",
        "[7] E. A. Lopez-Rojas and S. Axelsson, “PaySim: A financial mobile money simulator for fraud detection,” in Proc. European Modeling Symposium, 2016.",
        "[8] S. El Kafhali, M. Tayebi, and H. Sulimani, “An optimized deep learning approach for detecting fraudulent transactions,” Information, vol. 15, no. 4, p. 227, 2024.",
        "[9] T. Amarasinghe, A. Aponso, and N. Krishnarajah, “Critical analysis of machine learning based approaches for fraud detection in financial transactions,” in Proc. ICMLT, 2018.",
        "[10] F. A. Almarshad, M. Zakariah, G. A. Gashgari, and T. Vaiyapuri, “RABEM: risk-adaptive Bayesian ensemble model for fraud detection,” Scientific Reports, vol. 15, Art. no. 36796, 2025.",
        "[11] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. ICML, 2017.",
        "[12] Software artefact: https://github.com/Rounak2901/FRAUDGUARD",
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(r)
        set_run_font(run, size=9)

    doc.save(OUT_DOCS)
    doc.save(OUT_ROOT)
    print(f"Wrote {OUT_DOCS}")
    print(f"Wrote {OUT_ROOT}")


if __name__ == "__main__":
    build()
