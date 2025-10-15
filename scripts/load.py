"""
load.py

Utility functions for scanning directories and reading Excel files
used in the mAb developability project.

This module provides helper functions to:
    • Recursively scan a directory and find all files with a given extension.
    • Read multiple Excel files into pandas DataFrames.

Author: Eduardo Pacheco
"""

import os
from pathlib import Path
import pandas as pd


def scan_directory(directory_path, file_extension=".xlsx"):
    """
    Recursively scan a directory for files with a specific extension.

    Parameters
    ----------
    directory_path : str or Path
        Path to the directory to scan.
    file_extension : str, optional
        File extension to search for (default is ".xlsx").

    Returns
    -------
    list of str
        A list of relative file paths (relative to the base directory)
        for all files matching the given extension.

    Notes
    -----
    - Uses os.walk to recursively traverse subdirectories.
    - Returns an empty list if no files are found or if an error occurs.
    """
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
    """
    Read multiple Excel files into pandas DataFrames.

    Parameters
    ----------
    files_list : list of str or Path
        List of file paths to read.

    Returns
    -------
    list of pandas.DataFrame
        A list of DataFrames, one per file in the input list.

    Notes
    -----
    - Each file must be a readable Excel file (.xlsx).
    - Files are read using `pandas.read_excel()`.
    - Returns an empty list if no valid files are found.
    """
    dfs = []
    for file in files_list:
        try:
            df = pd.read_excel(file)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipping file '{file}' due to error: {e}")

    return dfs
