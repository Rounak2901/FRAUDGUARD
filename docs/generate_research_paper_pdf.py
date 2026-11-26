"""
Generate an original research-paper PDF for Fraud Guard.
Run: py -m docs.generate_research_paper_pdf
Output: docs/FraudGuard_Research_Paper.pdf
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    ListFlowable,
    ListItem,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "FraudGuard_Research_Paper.pdf"

NAVY = HexColor("#0f2744")
TEAL = HexColor("#0d9488")
LIGHT = HexColor("#f1f5f9")


def styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=16,
            leading=20,
            alignment=TA_CENTER,
            textColor=NAVY,
            spaceAfter=8,
        ),
        "authors": ParagraphStyle(
            "Authors",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "affil": ParagraphStyle(
            "Affil",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=HexColor("#334155"),
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=12,
            leading=15,
            textColor=NAVY,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Times-Bold",
            fontSize=11,
            leading=14,
            textColor=HexColor("#1e293b"),
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyJust",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=9.5,
            leading=12.5,
            alignment=TA_JUSTIFY,
            leftIndent=12,
            rightIndent=12,
            spaceAfter=6,
        ),
        "kw": ParagraphStyle(
            "Keywords",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=9,
            leading=12,
            leftIndent=12,
            rightIndent=12,
            spaceAfter=10,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=8.5,
            leading=11,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=10,
        ),
        "ref": ParagraphStyle(
            "Ref",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=8.5,
            leading=11,
            leftIndent=12,
            firstLineIndent=-12,
            spaceAfter=3,
        ),
        "foot": ParagraphStyle(
            "Foot",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=8,
            textColor=HexColor("#64748b"),
            alignment=TA_CENTER,
        ),
    }
    return s


def p(text: str, style):
    return Paragraph(text.replace("\n", " "), style)


def make_table(headers, rows, col_widths=None):
    data = [headers] + rows
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#94a3b8")),
                ("BACKGROUND", (0, 1), (-1, -1), LIGHT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, white]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(HexColor("#cbd5e1"))
    canvas.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(HexColor("#64748b"))
    canvas.drawString(2 * cm, 0.9 * cm, "Fraud Guard — Cost-Sensitive Bayesian Real-Time Fraud Detection")
    canvas.drawRightString(A4[0] - 2 * cm, 0.9 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    s = styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2 * cm,
        title="Cost-Sensitive Bayesian Neural Detection of Online Payment Fraud with Entity-Level Risk Updating",
        author="Rounak Dhall",
    )
    story = []

    story.append(
        p(
            "Cost-Sensitive Bayesian Neural Detection of Online Payment Fraud "
            "with Entity-Level Risk Updating and Real-Time Explainable Decisions",
            s["title"],
        )
    )
    story.append(p("Rounak Dhall", s["authors"]))
    story.append(
        p(
            "Department of Computer Science and Engineering (B.Tech.), Independent Research Prototype — Fraud Guard System<br/>"
            "Corresponding author: rounak2901@gmail.com &nbsp;|&nbsp; Code: https://github.com/Rounak2901/FRAUDGUARD",
            s["affil"],
        )
    )

    story.append(p("<b>Abstract</b>", s["h1"]))
    story.append(
        p(
            "Online payment fraud is rare, fast, and asymmetric in cost: missing a fraudulent transfer "
            "typically destroys far more value than incorrectly challenging a legitimate customer. "
            "Accuracy is therefore a misleading headline metric under extreme class imbalance. "
            "This paper presents <i>Fraud Guard</i>, an end-to-end detection architecture that couples "
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
            s["abstract"],
        )
    )
    story.append(
        p(
            "<b>Keywords:</b> payment fraud detection; class imbalance; PR-AUC; Bayesian neural networks; "
            "Monte Carlo Dropout; Beta–Bernoulli updating; cost-sensitive learning; anomaly detection; "
            "explainable AI; real-time scoring",
            s["kw"],
        )
    )

    # 1
    story.append(p("1. Introduction", s["h1"]))
    story.append(
        p(
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
            s["body"],
        )
    )
    story.append(
        p(
            "This work addresses those gaps with a unified prototype named Fraud Guard. The design goal is "
            "not merely to maximise a ranking metric, but to minimise expected operational loss under a "
            "configurable cost matrix while remaining interpretable to analysts and deployable on commodity "
            "CPU hardware. The contributions are:",
            s["body"],
        )
    )
    story.append(
        ListFlowable(
            [
                ListItem(
                    p(
                        "A time-respecting training protocol with robust amount scaling fit on the training "
                        "window only, and causal entity features that never peek into the future.",
                        s["body"],
                    ),
                    leftIndent=12,
                ),
                ListItem(
                    p(
                        "A Bayesian neural layer via Monte Carlo Dropout that yields predictive means, "
                        "uncertainty, and approximate credible intervals used to open a Review band.",
                        s["body"],
                    ),
                    leftIndent=12,
                ),
                ListItem(
                    p(
                        "Online Beta–Bernoulli entity-risk beliefs with mild temporal decay, updated from "
                        "analyst feedback and injected into both the ensemble score and the dashboard.",
                        s["body"],
                    ),
                    leftIndent=12,
                ),
                ListItem(
                    p(
                        "A cost-sensitive Approve/Review/Block engine with validation-set threshold search, "
                        "ablation of ensemble components, and a working real-time API with explainable reasons.",
                        s["body"],
                    ),
                    leftIndent=12,
                ),
            ],
            bulletType="1",
            start=1,
        )
    )

    # 2
    story.append(p("2. Related Work", s["h1"]))
    story.append(
        p(
            "Supervised fraud classifiers on anonymised credit-card ledgers are well studied. Classical "
            "baselines such as logistic regression, random forests, and gradient boosting remain strong "
            "references under careful validation. Deep architectures—including multilayer perceptrons, "
            "recurrent networks, and sequence models—have been explored for capturing temporal dependence, "
            "sometimes with Bayesian hyperparameter search to stabilise training under imbalance. Parallel "
            "streams investigate unsupervised or hybrid detectors when labels are delayed, for example "
            "isolation forests and autoencoders trained on genuine traffic.",
            s["body"],
        )
    )
    story.append(
        p(
            "Survey and critical analyses of financial fraud analytics emphasise that discrete rule checks "
            "on accounts or devices are insufficient against adaptive adversaries, and that supervised "
            "anomaly methods must be complemented by unsupervised cues. Recent Bayesian ensemble proposals "
            "combine multiple learners and reliability fusion to improve robustness on synthetic mobile-money "
            "streams. Our work differs in focus: rather than introducing another opaque high-capacity "
            "ensemble, we organise uncertainty, entity beliefs, anomaly evidence, and dollar costs into an "
            "operational decision policy with explicit review routing and live explainability.",
            s["body"],
        )
    )

    # 3
    story.append(p("3. Problem Formulation", s["h1"]))
    story.append(
        p(
            "Let each transaction be a feature vector <font face='Courier'>x</font> with amount "
            "<font face='Courier'>a</font> and binary label <font face='Courier'>y ∈ {0,1}</font> "
            "(1 = fraud). A scoring function returns a probability "
            "<font face='Courier'>p̂ = P(y=1|x)</font> and an uncertainty summary "
            "<font face='Courier'>u(x)</font>. Actions are drawn from "
            "<font face='Courier'>{Approve, Review, Block}</font>. We adopt a simple but transparent "
            "cost model: a false positive incurs a fixed friction cost "
            "<font face='Courier'>C_fp</font> (investigation or customer inconvenience), while a false "
            "negative loses approximately <font face='Courier'>a · C_fn</font>. Review incurs "
            "<font face='Courier'>C_review</font> and is preferred when <font face='Courier'>u(x)</font> "
            "is large or when <font face='Courier'>p̂</font> lies near the cost-optimal boundary. "
            "The primary ranking metric is PR-AUC on a chronologically held-out test window; secondary "
            "metrics include ROC-AUC, F2, Matthews correlation (MCC), Brier score, and expected "
            "calibration error (ECE).",
            s["body"],
        )
    )

    # 4
    story.append(p("4. Proposed System: Fraud Guard", s["h1"]))
    story.append(p("4.1 Data pipeline and leakage controls", s["h2"]))
    story.append(
        p(
            "Transactions are sorted by time and partitioned into contiguous 70%/15%/15% blocks for "
            "training, validation, and testing. Amount is log-transformed and robust-scaled with "
            "statistics estimated on the training block alone. Hour-of-day is encoded with sine/cosine "
            "harmonics. For streaming entity memory we maintain per-card rolling counts, amount moments, "
            "velocity, and time since last purchase using only past events for that entity.",
            s["body"],
        )
    )
    story.append(p("4.2 Supervised neural classifier", s["h2"]))
    story.append(
        p(
            "The core model is a three-hidden-layer multilayer perceptron (128–64–32 units) with batch "
            "normalisation, ReLU activations, and dropout. Training compares class-weighted binary "
            "cross-entropy against focal loss; early stopping monitors validation PR-AUC. Fixed seeds "
            "support reproducibility on CPU-only laptops.",
            s["body"],
        )
    )
    story.append(p("4.3 Uncertainty via Monte Carlo Dropout", s["h2"]))
    story.append(
        p(
            "Following the interpretation of dropout as approximate variational inference, we keep dropout "
            "active at test time and draw T stochastic forward passes (T = 40 in experiments). The mean "
            "probability estimates fraud likelihood; the empirical standard deviation proxies epistemic "
            "uncertainty. A normal approximation yields a 95% interval used in the user interface. "
            "Isotonic regression fitted on the validation window recalibrates probabilities before "
            "decisioning.",
            s["body"],
        )
    )
    story.append(p("4.4 Bayesian entity-risk updating", s["h2"]))
    story.append(
        p(
            "For each entity identifier we maintain a Beta(<font face='Courier'>α, β</font>) belief with "
            "a weakly informative prior matched to a low global fraud rate "
            "(e.g., <font face='Courier'>α0=1</font>, <font face='Courier'>β0=500</font>). Confirmed fraud "
            "increments <font face='Courier'>α</font>; confirmed genuine increments "
            "<font face='Courier'>β</font>. A daily multiplicative decay gently forgets stale evidence. "
            "The posterior mean enters the ensemble and is visualised live after analyst feedback.",
            s["body"],
        )
    )
    story.append(p("4.5 Anomaly channel and ensemble", s["h2"]))
    story.append(
        p(
            "An autoencoder trained exclusively on genuine training rows supplies a reconstruction-error "
            "anomaly score, min–max normalised using validation statistics. The final risk score is a "
            "convex combination of calibrated neural probability, entity posterior mean, and anomaly score. "
            "Mixture weights are selected by grid search on validation PR-AUC.",
            s["body"],
        )
    )
    story.append(p("4.6 Cost-sensitive decisions and explanations", s["h2"]))
    story.append(
        p(
            "On the validation window we sweep decision thresholds to minimise total monetary cost under "
            "the configured matrix, then freeze the threshold for the test window. High uncertainty or "
            "proximity to the boundary triggers Review. Local reasons are produced from a gradient-boosting "
            "surrogate TreeExplainer when available, otherwise from signed standardised deviations weighted "
            "by linear coefficients, and rendered as plain-language bullets for analysts.",
            s["body"],
        )
    )
    story.append(p("4.7 Serving prototype", s["h2"]))
    story.append(
        p(
            "A FastAPI service validates inputs, scores transactions, persists decisions, updates beliefs "
            "from feedback, and streams events over WebSocket. A React dashboard provides a live monitor, "
            "review queue, entity explorer, cost simulator, and manual “try-it” form for demonstrations.",
            s["body"],
        )
    )

    # 5
    story.append(p("5. Experimental Setup", s["h1"]))
    story.append(
        p(
            "Experiments were executed on a standard CPU laptop using Python, PyTorch, scikit-learn, and "
            "XGBoost. The feature schema follows the widely used anonymised credit-card fraud setting "
            "(time, amount, and PCA components). Because institutional ledgers cannot be redistributed, "
            "the quantitative tables in this manuscript correspond to a fully reproducible synthetic stream "
            "that matches the public schema and a similar fraud prevalence (~0.17%). The same code path "
            "accepts the original Kaggle credit-card file and a PaySim sample for entity trajectories when "
            "credentials are available. Classical baselines (logistic regression, random forest, XGBoost, "
            "isolation forest) share the identical chronological split and metric suite.",
            s["body"],
        )
    )

    # 6
    story.append(p("6. Results and Discussion", s["h1"]))
    story.append(p("6.1 Ranking performance", s["h2"]))
    story.append(
        p(
            "Table 1 summarises test-set ranking quality. Logistic regression remained a formidable baseline "
            "(PR-AUC 0.9324). The multilayer perceptron with weighted BCE reached 0.9265 PR-AUC and near-ceiling "
            "ROC-AUC, confirming that a compact network is competitive when leakage is controlled. XGBoost "
            "achieved 0.8026 PR-AUC with balanced precision and recall at the default 0.5 threshold. "
            "Isolation forest ranked poorly on PR-AUC, as expected for a purely unsupervised score under "
            "severe imbalance, yet remained useful as a complementary anomaly channel.",
            s["body"],
        )
    )

    story.append(
        make_table(
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
            col_widths=[3.2 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 1.8 * cm, 1.6 * cm, 1.6 * cm],
        )
    )
    story.append(
        p(
            "Table 1. Test metrics at threshold 0.5 unless noted. *Ensemble precision/recall at 0.5 are "
            "unstable because the learned mixture shifts mass; operational performance is reported after "
            "cost-based thresholding (Section 6.3).",
            s["caption"],
        )
    )

    story.append(p("6.2 Calibration and uncertainty quality", s["h2"]))
    story.append(
        p(
            "Raw MC-Dropout means were over-confident on this stream (ECE 0.1803). After isotonic "
            "calibration on validation probabilities, ECE fell to 0.0005. Average uncertainty was higher "
            "on erroneous predictions (0.139) than on correct ones (0.085), supporting the use of "
            "uncertainty to gate Review rather than forcing binary auto-decisions.",
            s["body"],
        )
    )

    story.append(p("6.3 Cost optimisation and ablation", s["h2"]))
    story.append(
        p(
            "Optimising the decision threshold on validation cost yielded a substantially lower test loss "
            "than the conventional 0.5 cut-off (1013.2 versus 5568.4 under the configured matrix). "
            "Ablation showed that adding the anomaly channel improved PR-AUC relative to the calibrated "
            "network alone (0.8908 versus 0.8216). Entity beliefs primarily benefit live adaptation after "
            "feedback; their offline contribution depends on how richly identifiers repeat in the sample.",
            s["body"],
        )
    )

    story.append(
        make_table(
            ["Configuration", "Test PR-AUC", "Notes"],
            [
                ["Calibrated NN only", "0.8216", "MC-Dropout + isotonic"],
                ["NN + entity posterior", "0.8040", "Soft Beta mean features"],
                ["NN + anomaly", "0.8908", "Autoencoder residual"],
                ["Full ensemble", "0.8908", "Grid weights on validation"],
                ["Cost @ thr=0.5", "5568.4*", "Higher operational loss"],
                ["Cost @ val-opt thr", "1013.2*", "Selected then frozen"],
            ],
            col_widths=[4.5 * cm, 3.0 * cm, 7.5 * cm],
        )
    )
    story.append(p("Table 2. Ablation and cost comparison (*cost in experiment currency units).", s["caption"]))

    story.append(p("6.4 Discussion", s["h2"]))
    story.append(
        p(
            "Two practical lessons emerge. First, PR-AUC and cost can disagree with thresholded F1: the "
            "MLP’s default 0.5 operating point was overly aggressive (high recall, low precision), yet "
            "ranking quality remained strong—reinforcing the need for cost-aware thresholding. Second, "
            "calibration and uncertainty are not optional decorations; they determine whether a model’s "
            "probability can safely drive money-moving actions or only a review queue. The serving "
            "prototype demonstrates that these ideas can coexist with interactive explanations without "
            "requiring a GPU cluster.",
            s["body"],
        )
    )

    # 7
    story.append(p("7. Limitations and Threats to Validity", s["h1"]))
    story.append(
        p(
            "Public PCA features limit semantic interpretability of attributions. Synthetic entity "
            "identifiers, when used, do not reproduce real multi-hop fraud rings. The reported numbers "
            "are tied to a reproducible synthetic draw unless replaced with licensed Kaggle extracts; "
            "external validity to any single bank remains unproven. The adversary is passive in our "
            "evaluation—no adaptive evasion is simulated. Latency measurements exclude wide-area network "
            "variance of production payment switches.",
            s["body"],
        )
    )

    # 8
    story.append(p("8. Conclusion and Future Work", s["h1"]))
    story.append(
        p(
            "Fraud Guard integrates calibrated Bayesian neural scoring, entity-level Beta risk memory, "
            "anomaly evidence, and cost-sensitive Approve/Review/Block policies into a transparent "
            "real-time prototype. Empirically, chronological evaluation, calibration, and threshold "
            "optimisation matter as much as raw classifier choice. Future work includes graph neural "
            "models over transaction networks, online drift detectors, adversarial robustness tests, "
            "and deployment-oriented MLOps with continuous feedback from confirmed cases.",
            s["body"],
        )
    )

    story.append(p("Acknowledgments", s["h1"]))
    story.append(
        p(
            "The author thanks open dataset curators whose public benchmarks enable comparable fraud "
            "research, and notes that this manuscript describes an academic prototype rather than a "
            "certified payment product.",
            s["body"],
        )
    )

    story.append(p("References", s["h1"]))
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
        "[11] N. Nicolicchia and colleagues’ broader surveys on imbalanced learning and payment risk (various venues); see also scikit-learn and PyTorch documentation for implementation baselines.",
        "[12] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. ICML, 2017.",
    ]
    for r in refs:
        story.append(p(r, s["ref"]))

    story.append(Spacer(1, 16))
    story.append(
        p(
            "Manuscript type: Original research / systems paper (academic prototype). "
            "Software artefact: https://github.com/Rounak2901/FRAUDGUARD",
            s["foot"],
        )
    )

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
