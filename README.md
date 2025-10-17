# 🧬 Therapeutic mAb Developability Prediction and Clustering

## Project Overview

This repository contains the full pipeline developed for **Milestone II** of the **Master of Applied Data Science (MADS)** program at the **University of Michigan**.

The goal of this project is to analyze a dataset of therapeutic monoclonal antibodies (mAbs) at various stages of clinical development — Phase II, Phase III, and approved — to evaluate their **developability characteristics** using both **supervised** and **unsupervised machine learning** techniques.

---

## 🔧 Workflow Overview

The analysis is structured into three major components, integrated within a unified Python workflow (`main.py`):

### 1. **Data Engineering & Preprocessing**
- Implemented via the scripts `load.py` and `preprocess.py`.
- Automatically scans the directory for Excel input files and merges them into a unified `DataFrame`.
- Cleans and standardizes assay values, handles missing entries, and constructs model-ready datasets.
- Integrates amino acid sequence features and optional **ESM-2 embeddings** for downstream analysis.

### 2. **Supervised Learning – Developability Prediction**
- Implemented in `supervised_learning.py` (by **Mengyao Li**).
- Builds and evaluates regression models (Random Forest, Gradient Boosting, MLP, SVR) to predict  
  the **“Slope for Accelerated Stability”**, a quantitative indicator of mAb developability.
- Includes:
  - Cross-validation with hyperparameter tuning (`GridSearchCV`)
  - Visualization of prediction vs. observation
  - Feature importance ranking and ablation studies
  - Sensitivity analyses for model parameters and dataset size

### 3. **Unsupervised Learning – Feature Space Clustering**
- Implemented in `unsupervised_learning.py` (by **Jared Fox**).
- Performs exploratory analysis of mAb feature space using:
  - Amino acid composition encoding
  - Dimensionality reduction via **PCA**
  - Clustering using **KMeans** and **Gaussian Mixture Models (GMM)**
- Generates comparative metrics (silhouette, ARI, BIC/AIC) to evaluate clustering robustness.
- Automatically saves CSV summaries and visualization plots (`results/` folder).

---

## 📂 Repository Structure
.
├── scripts/
│   ├── load.py                     # File scanning and Excel import utilities
│   ├── preprocess.py               # Data cleaning and model-ready dataset construction
│   ├── supervised_learning.py      # Regression models for stability prediction
│   └── unsupervised_learning.py    # Clustering and dimensionality reduction analysis
├── data/                           # Raw input data (Excel files)
├── results/                        # Generated results (plots, CSVs)
├── main.py                         # Central execution script
└── README.md

---

## 🧠 Project Highlights

- End-to-end **machine learning pipeline** for therapeutic antibody developability
- Modular, reproducible code structure aligned with professional ML workflows
- Visualization outputs include:
  - Model learning curves and residual plots  
  - Feature importance bar charts  
  - PCA and clustering visualizations (KMeans & GMM)
- Results exported automatically to `/results` for reproducibility

---

## 👥 Team Contributions

| Member              | Responsibilities                                                                                                     |
| :------------------ | :------------------------------------------------------------------------------------------------------------------- |
| **Eduardo Pacheco** | Data loading, cleaning, and engineering; integration of all scripts; pipeline orchestration (`main.py`).             |
| **Mengyao Li**      | Supervised learning pipeline development: model training, regression tuning, feature analysis, and interpretability. |
| **Jared Fox**       | Unsupervised learning pipeline: PCA dimensionality reduction, KMeans and GMM clustering, and sensitivity analyses.   |

---

## 🚀 Running the Project

**1. Environment setup:**
```bash
pip install -r requirements.txt
```

**2. Execute the workflow:**
```bash
python main.py
```

## 🧩 Future Work

- Integrate **ESM-2 embeddings** directly into the unsupervised pipeline to compare clustering against classical features.
- Extend supervised models with **ensemble stacking** (e.g., RF + GB + SVR) for improved prediction performance.
- Explore **explainable AI (XAI)** methods (e.g., SHAP) to interpret feature importance across antibody subclasses.
- **Deploy** trained models as a reproducible analysis dashboard or API endpoint.


📘 Developed as part of the **University of Michigan MADS Program – Milestone II Project (2025)**.