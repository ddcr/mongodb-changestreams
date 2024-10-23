__author__ = "Domingos Rodrigues"
__email__ = "domingos.rodrigues@inventvision.com.br"
__copyright__ = "Copyright (C) 2024 Invent Vision"
__license__ = "Strictly proprietary for Invent Vision."

import os
import shutil
from pathlib import Path, PureWindowsPath

import pandas as pd


def is_windows_path(path_str):
    """
    Determines if the given path string is in Windows format.

    Args:
        path_str (str): The path string to check.

    Returns:
        bool: True if the path is in Windows format, False otherwise.
    """
    try:
        # Attempt to parse the path using PureWindowsPath
        windows_path = PureWindowsPath(path_str)
        # Check if the path's drive attribute is not empty or if it contains backslashes
        return bool(windows_path.drive) or "\\" in path_str
    except Exception:
        return False


def assert_same_extensions(*args):
    exts = [os.path.splitext(path)[1] for path in args]
    if not (exts[1:] == exts[:-1]):
        raise OSError(f"Expected {str(args)} to have the same extensions")


def copy_file(inpath, outpath, check_ext=False):
    if check_ext and os.path.splitext(outpath)[1]:
        assert_same_extensions(inpath, outpath)

    basedir = Path(outpath)
    basedir.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy(inpath, outpath)
    except shutil.SameFileError:
        pass


def load_scrap_rank(filePath: str) -> pd.DataFrame:
    data = pd.read_csv(filePath, dtype=str)
    data = data.astype({"price_ranking": "int32"})
    return data


scrapRankDF = load_scrap_rank('classes_info.csv')
scrapRank = scrapRankDF.set_index('code')['name'].to_dict()
