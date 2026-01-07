import pandas as pd
from pathlib import Path
from typing import Optional, Union, Sequence, Literal, Tuple, Set, List
import logging

logger = logging.getLogger(__name__)

def load_dataframe(
    path: Optional[Union[str, Path]],
    *,
    name: str,
) -> pd.DataFrame:
    """
    Load a DataFrame from CSV or Parquet.
    Returns an empty DataFrame if path is None.
    Raises for unsupported suffixes.
    """
    if path is None:
        return pd.DataFrame()

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported {name} format: {path.suffix}")
    except Exception as e:
        raise RuntimeError(f"Failed to load {name} from {path}") from e

    if df.empty:
        logger.warning(f"Warning: The {name} dataframe is empty")

    return df


def save_dataframe(
    df: pd.DataFrame,
    path: Union[str, Path],
    *,
    index: bool = False,
) -> None:
    """
    Save a DataFrame to CSV or Parquet based on file suffix.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            df.to_csv(path, index=index)
        elif suffix == ".parquet":
            df.to_parquet(path, index=index)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    except Exception as e:
        raise RuntimeError(f"Failed to save to {path}") from e
