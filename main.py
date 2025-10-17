import os
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from scripts.load import scan_directory, read_files
from scripts.preprocess import create_clean_df, build_model_ready_from_merged
from scripts.supervised_learning import (
    train_and_eval,
    get_feature_importance,
    plot_feature_importance,
    ablation_analysis,
    sensitivity_analysis,
    data_size_sensitivity,
    ensure_results_dir,
)

from scripts.unsupervised_learning import run_unsupervised_analysis


def main():
    print("\nScanning for files...")
    directory = os.getcwd()
    data_files = scan_directory(directory, file_extension=".xlsx")
    print("\nFiles found:", data_files)

    print("\nprecprocessing files...")
    dfs = read_files(data_files)
    print("\nNumber of dataframes:", len(dfs))

    merged_df = create_clean_df(dfs)
    print("\nMerged dataframe shape:", merged_df.shape)

    model_ready_df = build_model_ready_from_merged(
        merged_df,
        target_col="Slope for Accelerated Stability",
        include_assays=False,
        include_esm=True,
        esm_model_name="esm2_t6_8M_UR50D",
    )
    print("\nFiles processed succesfully ✅")

    # ==============================================
    # --------Supervised Learning Workflow----------
    # ==============================================

    print("\nConducting supervised learning routine...")

    # 1. Extract X and y from the model_ready_df
    X, y = model_ready_df["X"], model_ready_df["y"]
    # Alternatively, load precomputed ESM embeddings
    # X = pd.read_csv("/home/lime1t/Python_sc/MADS/milestoneii/data/esm_embeddings_labeled.csv", index_col=0)
    # 2. Get rid of the outlier (too high value in y)
    # drop the row for rilotumumab
    # Get the index of the row for rilotumumab
    row_to_drop = 109  # X.index.get_loc('rilotumumab')
    X = X.drop(X.index[row_to_drop])
    y = merged_df["Slope for Accelerated Stability"].values
    y = np.delete(y, row_to_drop)

    # 3. Train/evaluate each requested model
    rf = train_and_eval(X, y, "rf")
    gb = train_and_eval(X, y, "gb")
    svr = train_and_eval(X, y, "svr")
    mlp = train_and_eval(X, y, "mlp")

    # 4. Feature importance and Ablation analysis
    # get feature importance and get the top 20 features
    fi, method = get_feature_importance(svr, X, y)
    fi_top = plot_feature_importance(svr, X, y, top_n=20)
    # Run ablation
    results = ablation_analysis(svr, X, y, ablation_steps=[0, 10, 20, 30, 40, 50])

    # 5. Sensitivity Analysis
    rf_param_grid = {"n_estimators": [50, 100, 200, 500], "max_depth": [1, 5, 10, 20]}

    rf_sensitivity = sensitivity_analysis(
        RandomForestRegressor,
        X,
        y,
        rf_param_grid,
        scoring="neg_root_mean_squared_error",
    )

    print(rf_sensitivity.head())

    svr_param_grid = {
        "C": [1e-4, 1e-3, 0.01, 0.1, 1],
        "gamma": [1e-4, 1e-3, 0.01, 0.1, 1],
    }

    svr_sensitivity = sensitivity_analysis(
        SVR, X, y, svr_param_grid, scoring="neg_root_mean_squared_error"
    )

    print(svr_sensitivity.head())

    # 6. Learning curve
    train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]  # Fractions of training data

    svr_learning_curve = data_size_sensitivity(
        svr,
        X,
        y,
        train_sizes=train_sizes,
        scoring="neg_root_mean_squared_error",
        cv=5,
        test_size=0.2,
    )

    print(svr_learning_curve)
    save_dir = ensure_results_dir(subfolder="supervised")

    # ==============================================
    # --------Unsupervised Learning Workflow--------
    # ==============================================
    print("\nConducting unsupervised learning routine...")
    unsup_results = run_unsupervised_analysis(merged_df, output_dir="results")
    print(unsup_results)


if __name__ == "__main__":
    main()
