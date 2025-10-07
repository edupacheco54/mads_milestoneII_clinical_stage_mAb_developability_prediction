import pandas as pd
import numpy as np
import esm
import torch

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


def embed_sequence_esm2(seq, model, alphabet, device="cpu"):
    if not isinstance(seq, str) or len(seq.strip()) == 0:
        return np.zeros(model.embed_dim, dtype=np.float32)

    batch_converter = alphabet.get_batch_converter()
    data = [("seq1", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        layer = model.num_layers
        results = model(batch_tokens, repr_layers=[layer], return_contacts=False)
        token_representations = results["representations"][layer]

    embedding = token_representations[0, 1:-1].mean(0).cpu().numpy().astype(np.float32)
    return embedding


def load_esm2_model(model_name="esm2_t6_8M_UR50D", device=None):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval().to(device)
    print(f"Loaded {model_name} on device: {device.upper()}")
    return model, alphabet, device


def build_model_ready_from_merged(
    merged_df,
    target_col="Slope for Accelerated Stability",
    seq_cols=("VH", "VL"),
    include_assays=False,
    impute_strategy="median",
    include_esm=True,
    esm_model_name="esm2_t6_8M_UR50D",
):
    if target_col in merged_df.columns:
        df = merged_df.dropna(subset=[target_col]).copy()
    else:
        df = merged_df.copy()

    # Classic engineered features
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

    # ESM-2 embeddings integration
    esm_X, esm_y = None, None
    if include_esm:
        print("🔷 Loading ESM-2 Model...")
        model, alphabet, device = load_esm2_model(esm_model_name)

        esm_embeddings = []
        for _, row in df.iterrows():
            vh_seq, vl_seq = row.get("VH", ""), row.get("VL", "")
            vh_emb = embed_sequence_esm2(vh_seq, model, alphabet, device)
            vl_emb = embed_sequence_esm2(vl_seq, model, alphabet, device)
            combined_emb = np.concatenate([vh_emb, vl_emb])
            esm_embeddings.append(combined_emb)

        esm_X = np.vstack(esm_embeddings)
        esm_y = y.copy() if y is not None else None

    return {
        "X": X,
        "y": y,
        "esm_X": esm_X,
        "esm_y": esm_y,
        "feature_columns": feature_columns,
        "supervised_df": df,
        "imputer": imputer,
        "scaler": scaler,
    }
