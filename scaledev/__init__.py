from pathlib import Path


def get_data_dir() -> Path:
    """Get path to data directory in repo."""
    return Path(__file__).parents[1] / "data"
