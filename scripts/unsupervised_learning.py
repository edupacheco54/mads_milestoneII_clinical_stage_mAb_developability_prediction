"""
unsupervised_learning.py

Helper utilities for clustering analysis and dimensionality reduction in the
mAb developability project.

This module provides:
    • Amino acid composition feature encoding
    • KMeans and GMM clustering pipelines with multiple scalers
    • PCA-based dimensionality reduction and sensitivity analysis
    • Visualization and summary outputs (CSV, PNG)

Author: Jared Fox (modularized by Eduardo Pacheco)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.mixture import GaussianMixture

AA20 = list("ACDEFGHIKLMNPQRSTVWY")


# ==============================================================
# --- Amino Acid Composition Encoder ---
# ==============================================================


def aa_composition(seq: str) -> np.ndarray:
    """
    Compute normalized amino acid composition vector for a sequence.
    Returns a 20-length array corresponding to standard amino acids.
    """
    if not isinstance(seq, str) or not seq:
        return np.zeros(len(AA20))

    seq = seq.upper()
    counts = np.zeros(len(AA20))
    for aa in seq:
        try:
            counts[AA20.index(aa)] += 1
        except ValueError:
            continue
    L = counts.sum()
    return counts / L if L > 0 else counts


# ==============================================================
# --- Unsupervised Learning Pipeline ---
# ==============================================================


def run_unsupervised_analysis(merged_df, output_dir="."):
    """
    Run clustering and dimensionality analysis on the merged dataset.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Combined dataset with VH, VL, and assay columns.
    output_dir : str
        Output directory for CSV and PNG results.

    Returns
    -------
    pd.DataFrame : comparison table of best clustering models.
    """

    # --- Ensure output directory exists ---
    output_dir = os.path.join("results", "unsupervised")
    os.makedirs(output_dir, exist_ok=True)

    df = merged_df.copy()
    id_col = "Name"

    # --- Sequence assembly ---
    df["Sequence"] = (df["VH"].fillna("") + df["VL"].fillna("")).replace("", np.nan)
    df = df.dropna(subset=["Sequence"]).reset_index(drop=True)

    # --- Amino acid composition features ---
    seq_features = np.vstack([aa_composition(s) for s in df["Sequence"]])
    seq_features = pd.DataFrame(
        seq_features, columns=[f"AAC_{aa}" for aa in AA20], index=df.index
    )

    # --- Biophysical assays ---
    candidate_assays = [
        "HIC Retention Time (Min)a",
        "SMAC Retention Time (Min)a",
        "CIC Retention Time (Min)",
        "Fab Tm by DSF (°C)",
        "Poly-Specificity Reagent (PSR) SMP Score (0-1)",
        "CSI-BLI Delta Response (nm)",
        "ELISA",
        "BVP ELISA",
        "HEK Titer (mg/L)",
        "Slope for Accelerated Stability",
    ]
    assay_cols = [c for c in candidate_assays if c in df.columns]
    assays = (
        df[assay_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(lambda x: x.median())
    )

    # --- Feature matrix ---
    X = pd.concat([seq_features, assays], axis=1)

    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
    }

    k_grid = list(range(2, min(8, len(df))))
    seeds = list(range(42, 52))
    fixed_n_list = [2, 5, 8, 10]
    evr_targets = [0.90, 0.95]
    records = []

    # ==============================================================
    # --- KMeans Sensitivity Analysis ---
    # ==============================================================

    for scaler_name, scaler_obj in scalers.items():
        Xs = scaler_obj.fit_transform(X)
        n_features = Xs.shape[1]

        # 1. No PCA
        _evaluate_scaling_strategy(
            Xs, k_grid, seeds, scaler_name, "no_pca", "-", n_features, records
        )

        # 2. Fixed-n PCA
        for n_req in fixed_n_list:
            n_comp = int(min(max(n_req, 2), n_features))
            scores = PCA(n_components=n_comp, random_state=42).fit_transform(Xs)
            _evaluate_scaling_strategy(
                scores,
                k_grid,
                seeds,
                scaler_name,
                "fixed_n",
                f"n={n_comp}",
                n_comp,
                records,
            )

        # 3. EVR-based PCA
        pca_full = PCA(n_components=min(50, n_features), random_state=42).fit(Xs)
        cumevr = np.cumsum(pca_full.explained_variance_ratio_)
        for tgt in evr_targets:
            m = max(2, int(np.argmax(cumevr >= tgt) + 1))
            Z = pca_full.transform(Xs)[:, :m]
            _evaluate_scaling_strategy(
                Z, k_grid, seeds, scaler_name, "evr", f"evr>={tgt:.2f}", m, records
            )

    sensitivity_df = pd.DataFrame.from_records(records).sort_values(
        ["Mean_Silhouette", "Mean_ARI_over_inits"], ascending=[False, False]
    )

    sensitivity_path = f"{output_dir}/cluster_sensitivity.csv"
    sensitivity_df.to_csv(sensitivity_path, index=False)
    print(f"[INFO] Saved: {sensitivity_path}")

    _plot_cluster_robustness(sensitivity_df, output_dir)

    # ==============================================================
    # --- Best Config + Visualization ---
    # ==============================================================
    best_cfg = _select_best_configuration(sensitivity_df)
    print("Top configuration:", best_cfg)

    comparison_df = _run_final_clustering(X, df, id_col, best_cfg, output_dir)
    return comparison_df


# ==============================================================
# --- Helper Subroutines ---
# ==============================================================
def _evaluate_scaling_strategy(
    X, k_grid, seeds, scaler_name, strategy, param, n_comp, records
):
    """Evaluate one scaling and PCA configuration."""
    for k in k_grid:
        silhouettes, run_labels = [], []
        for s in seeds:
            km = KMeans(n_clusters=k, n_init=20, random_state=s)
            labels = km.fit_predict(X)
            run_labels.append(labels)
            silhouettes.append(silhouette_score(X, labels))
        mean_sil = np.mean(silhouettes)
        std_sil = np.std(silhouettes)
        pair_aris = [
            adjusted_rand_score(run_labels[i], run_labels[j])
            for i in range(len(run_labels))
            for j in range(i + 1, len(run_labels))
        ]
        records.append(
            {
                "Scaler": scaler_name,
                "Strategy": strategy,
                "StrategyParam": param,
                "n_components_used": n_comp,
                "k": k,
                "Mean_Silhouette": mean_sil,
                "STD_Silhouette": std_sil,
                "Mean_ARI_over_inits": np.mean(pair_aris),
                "STD_ARI_over_inits": np.std(pair_aris),
            }
        )


def _plot_cluster_robustness(sensitivity_df, output_dir):
    """Plot mean silhouette vs k across all scalers and PCA strategies."""
    agg = (
        sensitivity_df.groupby("k")[["Mean_Silhouette", "STD_Silhouette"]]
        .agg(Mean_Sil=("Mean_Silhouette", "mean"), Std_Sil=("STD_Silhouette", "mean"))
        .reset_index()
    )
    plt.figure(figsize=(8, 5))
    plt.errorbar(agg["k"], agg["Mean_Sil"], yerr=agg["Std_Sil"], fmt="o-")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette (mean ± avg std)")
    plt.title("Cluster Robustness vs k")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_sensitivity.png", dpi=150)
    plt.close()


def _select_best_configuration(sensitivity_df):
    """Select the top configuration based on silhouette and ARI."""
    best_row = sensitivity_df.iloc[0]
    return {
        "Scaler": str(best_row["Scaler"]),
        "Strategy": str(best_row["Strategy"]),
        "StrategyParam": str(best_row["StrategyParam"]),
        "n_components_used": int(best_row["n_components_used"]),
        "k": int(best_row["k"]),
    }


def _run_final_clustering(X, df, id_col, best_cfg, output_dir):
    """Run final clustering (KMeans + GMM) with visualization."""
    scaler_map = {
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "RobustScaler": RobustScaler,
    }
    ScalerClass = scaler_map[best_cfg["Scaler"]]
    Xs = ScalerClass().fit_transform(X)

    n_full = min(10, X.shape[1])
    pca_full = PCA(n_components=n_full, random_state=42)
    scores_full = pca_full.fit_transform(Xs)
    strategy = best_cfg["Strategy"]

    if strategy == "no_pca":
        Z = Xs
        viz = PCA(n_components=2, random_state=42).fit_transform(Xs)
    elif strategy == "fixed_n":
        n_keep = max(2, min(best_cfg["n_components_used"], X.shape[1]))
        Z, viz = scores_full[:, :n_keep], scores_full[:, :2]
    else:
        Z, viz = scores_full[:, : best_cfg["n_components_used"]], scores_full[:, :2]

    # --- KMeans ---
    k_best = best_cfg["k"]
    km = KMeans(n_clusters=k_best, n_init=20, random_state=42)
    labels_km = km.fit_predict(Z)
    sil_km = silhouette_score(Z, labels_km)

    # --- GMM ---
    k_grid = range(2, 8)
    bic_vals, aic_vals = [], []
    for k in k_grid:
        gmm = GaussianMixture(
            n_components=k, covariance_type="full", n_init=10, random_state=42
        ).fit(Z)
        bic_vals.append(gmm.bic(Z))
        aic_vals.append(gmm.aic(Z))
    best_k_gmm = list(k_grid)[np.argmin(bic_vals)]
    best_gmm = GaussianMixture(
        n_components=best_k_gmm, covariance_type="full", n_init=10, random_state=42
    ).fit(Z)
    labels_gmm = best_gmm.predict(Z)
    sil_gmm = silhouette_score(Z, labels_gmm)

    # --- Comparison summary ---
    comparison = pd.DataFrame(
        [
            {
                "Family": "KMeans",
                "Scaler": best_cfg["Scaler"],
                "Dimensionality": f"{strategy} ({best_cfg['StrategyParam']})",
                "k": k_best,
                "Silhouette": sil_km,
                "Model Selection": "—",
            },
            {
                "Family": "GMM(full)",
                "Scaler": best_cfg["Scaler"],
                "Dimensionality": f"{strategy} ({best_cfg['StrategyParam']})",
                "k": best_k_gmm,
                "Silhouette": sil_gmm,
                "Model Selection": f"BIC={np.min(bic_vals):.1f}",
            },
        ]
    )

    comparison.to_csv(f"{output_dir}/unsup_comparison_table.csv", index=False)
    print(f"[INFO] Saved comparison table at {output_dir}/unsup_comparison_table.csv")

    # --- Visualization ---
    plt.scatter(viz[:, 0], viz[:, 1], c=labels_km, s=25)
    plt.title(f"KMeans Clustering (k={k_best}) — Silhouette={sil_km:.3f}")
    plt.savefig(f"{output_dir}/top_config_kmeans.png", dpi=150)
    plt.close()

    plt.plot(list(k_grid), bic_vals, marker="o", label="BIC")
    plt.plot(list(k_grid), aic_vals, marker="o", label="AIC")
    plt.xlabel("k")
    plt.ylabel("Information Criterion")
    plt.title("GMM Model Selection")
    plt.legend()
    plt.savefig(f"{output_dir}/gmm_bic_aic.png", dpi=150)
    plt.close()

    return comparison
