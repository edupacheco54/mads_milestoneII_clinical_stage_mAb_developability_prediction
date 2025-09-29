import os

from scripts.load import scan_directory, read_files
from scripts.preprocess import create_clean_df, build_model_ready_from_merged


def main():
    directory = os.getcwd()
    data_files = scan_directory(directory, file_extension=".xlsx")
    dfs = read_files(data_files)
    merged_df = create_clean_df(dfs)
    model_ready_df = build_model_ready_from_merged(
        merged_df,
        target_col="Slope for Accelerated Stability",
        seq_cols=("VH", "VL"),
        include_assays=False,
        impute_strategy="median",
    )

    return model_ready_df


if __name__ == "__main__":
    main()
