# Model artifacts

Trained weights and metadata are written here by `py -m src.train` (later phases):

- `baselines/` — Logistic Regression, Random Forest, XGBoost, Isolation Forest
- `nn_mlp.pt` — deterministic MLP
- `bnn_mc_dropout.pt` — MC-Dropout Bayesian NN
- `autoencoder.pt` — reconstruction anomaly model
- `ensemble_weights.json` — stacking / grid-search weights
- `shap_explainer.joblib` — cached explainer
- `decision_thresholds.json` — cost-optimal thresholds
- `registry.json` — model version metadata
