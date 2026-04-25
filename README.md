# Machine Learning-Based Prediction of Nanoparticle Tumour Biodistribution

BSc Computer Science final-year project, Cardiff University.

## What this is

An ensemble ML pipeline (SciBERT text embeddings + neural network + XGBoost)
with SHAP interpretability and a Streamlit interface, built to predict
nanoparticle tumour-targeting efficiency from physicochemical properties
and free-text descriptions.

The dissertation report is in this repo (`Final_Report.pdf`). Source code
is on local infrastructure and can be made available on request.

## Key components

- **Data preprocessing:** robust parsing of heterogeneous Excel data
  (handles formats like `TEM:37.5`, `IV:4.06 (21g BW)`), schema detection,
  median/mode imputation, Parquet outputs.
- **Text embeddings:** SciBERT (`allenai/scibert_scivocab_uncased`) producing
  768-dimensional vectors from free-text nanoparticle descriptions.
- **Models:** feedforward NN (128→64→1, ReLU, Adam, early stopping) and
  XGBoost (max_depth=6, lr=0.1) combined as a simple-average ensemble.
- **Interpretability:** SHAP TreeExplainer over the XGBoost component for
  global and local feature attributions.
- **Interface:** Streamlit app with tabs for model overview, SHAP
  explanations, and a simulated collaborative-retraining workflow.

## Results

On an 80/20 split of the Nano-Tumor Database:

| Setting                              | RMSE   | R²     |
|--------------------------------------|--------|--------|
| Baseline (full data)                 | 3.444  | 0.031  |
| Stripped dataset (~85%)              | 3.147  | 0.012  |
| Stripped + simulated contribution    | 2.675  | 0.013  |

## Honest limitations

- R² values are low. The dataset is small (~hundreds of usable rows),
  highly heterogeneous across labs, and biologically complex. The model
  beats the mean only modestly — the contribution of this project is
  the engineering and interpretability scaffolding, not strong predictive
  accuracy.
- "Collaborative retraining" simulates federated-style updates by holding
  out a subset and retraining on the union; it does not implement true
  federated learning protocols.
- Many biological variables (immune response, tumour heterogeneity,
  perfusion) are absent from the dataset.

## Stack

Python, PyTorch/Keras, XGBoost, scikit-learn, transformers (SciBERT via
sentence-transformers), SHAP, Streamlit, pandas, NumPy, Parquet.
