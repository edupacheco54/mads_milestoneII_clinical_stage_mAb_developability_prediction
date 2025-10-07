import os

from scripts.load import scan_directory, read_files
from scripts.preprocess import create_clean_df, build_model_ready_from_merged


def main():
    directory = os.getcwd()
    data_files = scan_directory(directory, file_extension=".xlsx")
    print("Files found:", data_files)

    dfs = read_files(data_files)    
    print("Number of dataframes:", len(dfs))

    merged_df = create_clean_df(dfs)
    print("Merged dataframe shape:", merged_df.shape)

    
    model_ready_df = build_model_ready_from_merged(
        merged_df,
        target_col="Slope for Accelerated Stability",
        include_assays=False,
        include_esm=True,
        esm_model_name="esm2_t6_8M_UR50D"
    )

    return model_ready_df

if __name__ == "__main__":
    main()
