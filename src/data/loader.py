from pathlib import Path
import pandas as pd


def load_data(
    file_path: str | Path,
    sep: str = ";",
    usecols=None,
    dtype=None,
) -> pd.DataFrame:
    """
    Load raw tabular pose data with optional column and dtype selection.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found at: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=sep,
        usecols=usecols,
        dtype=dtype,
    )

    print(f"Data loaded successfully from: {file_path}")
    print(f"Data shape: {df.shape}")
    print(df.dtypes.head())
    return df