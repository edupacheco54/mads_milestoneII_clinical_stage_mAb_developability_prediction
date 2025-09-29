import os
import pandas as pd

from pathlib import Path


def scan_directory(directory_path, file_extension=".xlsx"):
    data_files = []
    try:
        directory = Path(directory_path)
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(file_extension):
                    file_path = Path(root) / file
                    relative_path = str(file_path.relative_to(directory))
                    data_files.append(relative_path)

        return data_files
    except Exception as e:
        print(f"❌ Error scanning directory: {e}")
        return []


def read_files(files_list):
    dfs = []
    for file in files_list:
        df = pd.read_excel(file)
        dfs.append(df)

    return dfs
