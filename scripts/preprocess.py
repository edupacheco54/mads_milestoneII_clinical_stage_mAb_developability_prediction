import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def create_clean_df(list_of_dfs):
    merged_df = list_of_dfs[0]
    for i in range(1, len(list_of_dfs)):
        merged_df = pd.merge(merged_df, list_of_dfs[i], on="Name", how="inner")

    merged_df = merged_df.dropna(axis=1, how="all")

    return merged_df


# Kyte-Doolitle (GRAVY) table
KD = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

HYDROPHOBIC = set("AILMVFWYV")
AROMATIC = set("FWY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
POLAR = set("STNQ")


def aa_counts(seq):
    counts = {aa: 0 for aa in KD.keys()}
    if isinstance(seq, str):
        for ch in seq.upper():
            if ch in counts:
                counts[ch] += 1

    return counts


def seq_features(seq, prefix):
    counts = aa_counts(seq)
    L = sum(counts.values())
    feats = {f"{prefix}_len": float(L)}

    if L == 0:
        for k in [
            "hydrophobic_frac",
            "aromatic_frac",
            "positive_frac",
            "negative_frac",
            "polar_frac",
            "kd_gravy",
        ]:
            feats[f"{prefix}_{k}"] = 0.0
        for aa in KD.keys():
            feats[f"{prefix}_comp_{aa}"] = 0.0

        return feats

    feats[f"{prefix}_hydrophobic_frac"] = (
        sum(counts[a] for a in HYDROPHOBIC if a in counts) / L
    )
    feats[f"{prefix}_aromatic_frac"] = (
        sum(counts[a] for a in AROMATIC if a in counts) / L
    )
    feats[f"{prefix}_positive_frac"] = (
        sum(counts[a] for a in POSITIVE if a in counts) / L
    )
    feats[f"{prefix}_negative_frac"] = (
        sum(counts[a] for a in NEGATIVE if a in counts) / L
    )
    feats[f"{prefix}_polar_frac"] = sum(counts[a] for a in POLAR if a in counts) / L
    feats[f"{prefix}_kd_gravy"] = sum(KD[a] * counts[a] for a in KD.keys()) / L

    for aa in KD.keys():
        feats[f"{prefix}_comp_{aa}"] = counts[aa] / L

    return feats


def engineer_sequence_features(df, seq_cols=("VH", "VL")):
    rows = []
    for _, row in df.iterrows():
        feats = {}
        for col in seq_cols:
            feats.update(seq_features(row.get(col, None), col))
        rows.append(feats)
    feat_df = pd.DataFrame(rows, index=df.index)
    return feat_df


def build_model_ready_from_merged(
    merged_df,
    target_col="Slope for Accelerated Stability",
    seq_cols=("VH", "VL"),
    include_assays=False,
    impute_strategy="median",
):
    if target_col in merged_df.columns:
        df = merged_df.dropna(subset=[target_col]).copy()
    else:
        df = merged_df.copy()

    seq_feats = engineer_sequence_features(df, seq_cols=seq_cols)

    if include_assays:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)
        assay_feats = df[num_cols].copy()
    else:
        assay_feats = pd.DataFrame(index=df.index)

    X_raw = pd.concat([seq_feats, assay_feats], axis=1)
    feature_columns = X_raw.columns.tolist()
    y = df[target_col].values if target_col in df.columns else None

    imputer = SimpleImputer(strategy=impute_strategy)
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X_raw)
    X_scaled = scaler.fit_transform(X_imputed)
    X = pd.DataFrame(X_scaled, index=df.index, columns=feature_columns)

    return {
        "X": X,
        "X_pca_input": X.copy(),
        "y": y,
        "feature_columns": feature_columns,
        "supervised_df": df,
        "imputer": imputer,
        "scaler": scaler,
    }
