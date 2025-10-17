"""
supervised_learning.py

Core utilities for training, evaluating, and interpreting supervised learning models
in the mAb developability project.

This module provides:
    • Functions to train and cross-validate multiple regression models (RF, GB, MLP, SVR)
    • Visualization tools for predictions, residuals, and learning curves
    • Model interpretability via feature importance and ablation analysis
    • Sensitivity analyses for hyperparameters and dataset size

Author: Mengyao Li
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    KFold,
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    root_mean_squared_error,
    mean_squared_error,
    r2_score,
)
from sklearn.inspection import permutation_importance
from sklearn.base import clone

from pathlib import Path


def ensure_results_dir(save_dir="results", subfolder=None):
    """Ensure the output directory exists (optionally with subfolder)."""
    path = Path(save_dir)
    if subfolder:
        path = path / subfolder
    path.mkdir(parents=True, exist_ok=True)
    return path


# ----------  plot functions for regression models  ----------
def plot_pred_vs_obs(y_train, y_train_pred, y_test, y_test_pred, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_train, y_train_pred, c="blue", label="Train", alpha=0.6)
    plt.scatter(y_test, y_test_pred, c="red", label="Test", alpha=0.6)
    minv, maxv = min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())
    plt.plot([minv, maxv], [minv, maxv], "k--")
    rmse = root_mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    plt.title(f"{title}\nTest RMSE={rmse:.2f}, R²={r2:.2f}")
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.legend()
    plt.tight_layout()
    # plt.show()


def plot_residuals(y_train, y_train_pred, y_test, y_test_pred, title):
    plt.figure(figsize=(7, 4))
    plt.scatter(
        y_train_pred, y_train - y_train_pred, c="blue", label="Train", alpha=0.6
    )
    plt.scatter(y_test_pred, y_test - y_test_pred, c="red", label="Test", alpha=0.6)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Obs – Pred)")
    plt.title(f"{title} Residuals")
    plt.legend()
    plt.tight_layout()
    # plt.show()


# ----------  MODEL TRAIN / EVAL  ----------
def train_and_eval(X, y, model_name):
    """Train and evaluate regression models, saving outputs to /results."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=41
    )

    # === Model selection ===
    if model_name == "rf":
        model = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid={
                "n_estimators": [10, 50, 200, 500],
                "max_depth": [None, 5, 10, 20],
            },
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
        )
    elif model_name == "gb":
        model = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_grid={
                "n_estimators": [100, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
            },
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
        )
    elif model_name == "mlp":
        model = GridSearchCV(
            MLPRegressor(max_iter=500, random_state=42),
            param_grid={
                "hidden_layer_sizes": [(64, 32), (128, 64), (256, 128)],
                "alpha": [1e-2, 0.1],
                "learning_rate_init": [1e-3, 0.01, 0.1],
            },
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
        )
    elif model_name == "svr":
        model = GridSearchCV(
            SVR(),
            param_grid={
                "kernel": ["rbf"],
                "C": [1e-2, 0.1, 1.0, 10.0],
                "gamma": ["scale", "auto", 1e-1, 1, 10],
                "epsilon": [1e-3, 0.01, 0.1],
            },
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
        )
    else:
        raise ValueError("Choose model_name from: rf, gb, mlp, svr")

    # === Fit model ===
    model.fit(X_train, y_train)
    print(f"\n=== {model_name.upper()} ===")
    print("Best Params:", model.best_params_)
    print("Best Score:", -model.best_score_)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(
        f"Train RMSE: {root_mean_squared_error(y_train, y_train_pred):.3f}, "
        f"Train R²: {r2_score(y_train, y_train_pred):.3f}"
    )
    print(
        f"Test  RMSE: {root_mean_squared_error(y_test, y_test_pred):.3f}, "
        f"Test  R²: {r2_score(y_test, y_test_pred):.3f}"
    )

    # === Visualization & Saving ===
    save_dir = ensure_results_dir(subfolder="supervised")

    # Predicted vs Observed
    plot_pred_vs_obs(y_train, y_train_pred, y_test, y_test_pred, model_name.upper())
    plt.savefig(
        save_dir / f"{model_name}_pred_vs_obs.png", dpi=150, bbox_inches="tight"
    )

    # Residuals
    plot_residuals(y_train, y_train_pred, y_test, y_test_pred, model_name.upper())
    plt.savefig(save_dir / f"{model_name}_residuals.png", dpi=150, bbox_inches="tight")

    # === Save Metrics ===
    metrics = {
        "Model": [model_name],
        "Best_Params": [model.best_params_],
        "Train_RMSE": [root_mean_squared_error(y_train, y_train_pred)],
        "Train_R2": [r2_score(y_train, y_train_pred)],
        "Test_RMSE": [root_mean_squared_error(y_test, y_test_pred)],
        "Test_R2": [r2_score(y_test, y_test_pred)],
    }
    pd.DataFrame(metrics).to_csv(save_dir / f"{model_name}_metrics.csv", index=False)

    print(f"Results saved to: {save_dir.resolve()}")

    return model.best_estimator_


def get_feature_importance(model, X, y):
    """Return feature importances or permutation importances for any sklearn model."""

    # If model is a GridSearchCV or pipeline, unwrap it
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_
    if hasattr(model, "named_steps"):
        last_step = list(model.named_steps.values())[-1]
    else:
        last_step = model

    # 1. Tree-based models
    if hasattr(last_step, "feature_importances_"):
        importance = last_step.feature_importances_
        method = "built-in"

    # 2. Linear SVR
    elif hasattr(last_step, "coef_"):
        importance = abs(last_step.coef_.ravel())
        method = "coefficients"

    # 3. All others → use permutation importance
    else:
        result = permutation_importance(
            model, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )
        importance = result.importances_mean
        method = "permutation"

    fi = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(
        by="Importance", ascending=False
    )

    print(f"Feature importance computed using: {method}")
    return fi, method


def plot_feature_importance(model, X, y, top_n=15):
    """Plot feature importances for any sklearn model."""
    fi, method = get_feature_importance(model, X, y)
    fi_top = fi.head(top_n)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=fi_top, x="Feature", y="Importance", palette="viridis")
    plt.title(f"Feature Importance ({method}) — top {top_n}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    # Rotate X-axis labels for better visibility
    plt.xticks(rotation=45, ha="right")  # Rotate labels 45 degrees, align right
    plt.tight_layout()
    # plt.show()

    # === Save outputs ===
    save_dir = ensure_results_dir(subfolder="supervised")
    fi.to_csv(
        save_dir / f"{model.__class__.__name__}_feature_importance.csv", index=False
    )
    plt.savefig(
        save_dir / f"{model.__class__.__name__}_feature_importance.png",
        dpi=150,
        bbox_inches="tight",
    )

    return fi_top


def ablation_analysis(
    model,
    X,
    y,
    ablation_steps=[0, 10, 20, 30, 40, 50],
    cv_folds=5,
    scoring="neg_root_mean_squared_error",
):
    """Perform ablation study by removing least important features stepwise."""
    fi, method = get_feature_importance(model, X, y)
    n_features = len(fi)
    results = []

    print(f"\nPerforming Ablation Analysis ({method} importance)...")

    for pct in ablation_steps:
        # keep top (100 - pct)% features
        n_keep = max(1, int(n_features * (1 - pct / 100)))
        top_features = fi["Feature"].iloc[:n_keep].tolist()

        X_sub = X[top_features]
        model_clone = clone(model)

        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(
            model_clone, X_sub, y, cv=cv, scoring=scoring, n_jobs=-1
        )
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        results.append(
            {
                "Removed_%": pct,
                "Remaining_Features": n_keep,
                "Mean_Score": mean_score,
                "Std_Score": std_score,
            }
        )

        print(
            f"Removed {pct}% | Remaining: {n_keep} | {scoring}: {mean_score:.4f} ± {std_score:.4f}"
        )

    df_results = pd.DataFrame(results)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        df_results["Removed_%"],
        -df_results["Mean_Score"],
        yerr=df_results["Std_Score"],
        fmt="-o",
        capsize=4,
    )
    plt.xlabel("Percentage of Least Important Features Removed")
    plt.ylabel("RMSE (lower = better)")
    plt.title("Ablation Analysis (Feature Importance)")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

    # === Save outputs ===
    save_dir = ensure_results_dir(subfolder="supervised")
    df_results.to_csv(save_dir / "ablation_analysis.csv", index=False)
    plt.savefig(save_dir / "ablation_analysis.png", dpi=150, bbox_inches="tight")

    return df_results


def sensitivity_analysis(
    model_class,
    X,
    y,
    param_grid,
    scoring="neg_root_mean_squared_error",
    cv=5,
    random_state=42,
):
    """
    Perform sensitivity analysis over 1D and 2D hyperparameter grid.

    Parameters
    ----------
    model_class : sklearn estimator class
        e.g., RandomForestRegressor, SVR
    X, y : pd.DataFrame
        Features and target
    param_grid : dict
        Dictionary with one or two hyperparameters and their value ranges
    scoring : str
        Scoring metric for cross-validation
    cv : int
        Number of CV folds
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    results_df : pd.DataFrame
        Mean and std of metric for each parameter combination
    """

    # Prepare grid
    keys = list(param_grid.keys())
    if len(keys) not in [1, 2]:
        raise ValueError("param_grid must contain 1 or 2 hyperparameters.")

    results = []

    # Grid search manually (no GridSearchCV)
    for vals in np.array(np.meshgrid(*param_grid.values())).T.reshape(-1, len(keys)):
        params = dict(zip(keys, vals))
        model = (
            model_class(random_state=random_state, **params)
            if "random_state" in model_class().get_params()
            else model_class(**params)
        )

        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        mean_score = -scores.mean() if scoring.startswith("neg_") else scores.mean()
        std_score = scores.std()

        result = {k: v for k, v in params.items()}
        result["Mean_Score"] = mean_score
        result["Std_Score"] = std_score
        results.append(result)

    results_df = pd.DataFrame(results)

    # Plot results
    fig = plt.figure(figsize=(14, 5))
    keys = list(param_grid.keys())

    # ---- 1D subplots ----
    for i, key in enumerate(keys):
        ax = fig.add_subplot(1, len(keys) + (len(keys) == 2), i + 1)
        means = results_df.groupby(key)["Mean_Score"].mean().reset_index()
        stds = results_df.groupby(key)["Std_Score"].mean().reset_index()
        ax.errorbar(
            means[key], means["Mean_Score"], yerr=stds["Std_Score"], fmt="-o", capsize=4
        )
        ax.set_xlabel(f"{key} log")
        ax.set_xscale("log")
        ax.set_ylabel("Mean RMSE")
        ax.set_title(f"Sensitivity: {key}")
        ax.grid(True)

    # ---- 3D plot if 2 params ----
    if len(keys) == 2:
        ax3d = fig.add_subplot(1, 3, 3, projection="3d")

        # Transform X and Y to log scale
        Xv, Yv = np.meshgrid(param_grid[keys[0]], param_grid[keys[1]])
        Xv_log = np.log10(Xv)
        Yv_log = np.log10(Yv)

        Z = np.round(
            results_df.pivot(
                index=keys[1], columns=keys[0], values="Mean_Score"
            ).values,
            2,
        )
        surf = ax3d.plot_surface(Xv_log, Yv_log, Z, cmap="viridis", alpha=0.8)

        # Set axis labels and ticks
        ax3d.set_xlabel(f"log({keys[0]})")
        ax3d.set_ylabel(f"log({keys[1]})")
        ax3d.set_zlabel("Mean RMSE")
        ax3d.set_title("Combined Parameter Sensitivity")
        fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10, pad=0.2, label="Mean RMSE")

    plt.tight_layout()
    # plt.show()

    # === Save outputs ===
    save_dir = ensure_results_dir(subfolder="supervised")
    results_df.to_csv(save_dir / "sensitivity_analysis.csv", index=False)
    plt.savefig(save_dir / "sensitivity_analysis.png", dpi=150, bbox_inches="tight")

    return results_df


def data_size_sensitivity(
    model,
    X,
    y,
    train_sizes,
    scoring="neg_root_mean_squared_error",
    cv=5,
    test_size=0.2,
    random_state=42,
):
    """
    Evaluate model performance as a function of training data size (randomly sampled subsets).
    Works with pandas DataFrames or NumPy arrays.
    Keeps the same test set across all runs.
    """
    rng = np.random.RandomState(random_state)

    # Fix test set once
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results = []
    n_samples = X_train_full.shape[0]

    for size in train_sizes:
        # Calculate absolute subset size
        subset_size = int(size * n_samples) if size <= 1 else int(size)
        subset_size = min(subset_size, n_samples)
        print("subset size:", subset_size)

        # Skip if subset smaller than CV folds
        if subset_size < cv:
            print(f"⚠️ Skipping size={subset_size}: fewer samples than CV folds ({cv})")
            continue

        # Randomly sample subset of training data
        subset_idx = rng.choice(n_samples, subset_size, replace=False)

        # Handle both pandas and numpy indexing
        if hasattr(X_train_full, "iloc"):
            X_train_sub = X_train_full.iloc[subset_idx]
        else:
            X_train_sub = X_train_full[subset_idx]

        if hasattr(y_train_full, "iloc"):
            y_train_sub = y_train_full.iloc[subset_idx]
        else:
            y_train_sub = y_train_full[subset_idx]

        # --- Cross-validation on subset ---
        scores = cross_val_score(
            model, X_train_sub, y_train_sub, cv=cv, scoring=scoring, n_jobs=-1
        )
        rmsecv_mean = -scores.mean() if scoring.startswith("neg_") else scores.mean()
        rmsecv_std = scores.std()

        # --- Fit model on the subset ---
        model.fit(X_train_sub, y_train_sub)

        # --- Predict on the fixed test set ---
        y_pred = model.predict(X_test)
        rmsep = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append(
            {
                "Train_Size": subset_size,
                "Train_Fraction": size,
                "RMSECV_Mean": rmsecv_mean,
                "RMSECV_Std": rmsecv_std,
                "RMSEP": rmsep,
            }
        )

    results_df = pd.DataFrame(results)

    # --- Plot results ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        results_df["Train_Fraction"],
        results_df["RMSECV_Mean"],
        yerr=results_df["RMSECV_Std"],
        fmt="-o",
        label="RMSECV (5-fold)",
        capsize=4,
    )
    ax.plot(
        results_df["Train_Fraction"],
        results_df["RMSEP"],
        "-s",
        label="RMSEP (Test)",
        alpha=0.8,
    )
    ax.set_xlabel("Training Data Size percentage")
    ax.set_ylabel("RMSE")
    ax.set_title("Impact of Training Data Size on Model Performance")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    # plt.show()

    # === Save outputs ===
    save_dir = ensure_results_dir(subfolder="supervised")
    results_df.to_csv(save_dir / "data_size_sensitivity.csv", index=False)
    plt.savefig(save_dir / "data_size_sensitivity.png", dpi=150, bbox_inches="tight")

    return results_df
