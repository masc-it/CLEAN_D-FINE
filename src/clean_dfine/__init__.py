from pathlib import Path


def get_data_dir():
    return Path(__file__).parent.parent.parent.expanduser().absolute() / "data"