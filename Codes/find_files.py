import glob
from pathlib import Path


def files():
    file_paths = glob.glob("../Data/*.csv")
    file_names = [Path(path).stem for path in file_paths]

    return file_names
