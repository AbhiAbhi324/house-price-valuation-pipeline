import argparse
from pathlib import Path    

import pandas as pd 
import numpy as np



ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

RAW_FILE_NAME ="AmesHousing.csv"


def find_raw_file(raw_dir: Path = RAW_DIR) -> Path:
        filename=RAW_FILE_NAME
        file_path = raw_dir / filename
        if file_path.exists():
            print(f"Found raw data file: {file_path}")
            return file_path
        raise FileNotFoundError("No raw data file found.")

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^\w\s]", "")
    return df

def load_raw_data(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else find_raw_file()
    df = pd.read_csv(path)
    return normalize_column_names(df)   






def main() -> None:
    parser = argparse.ArgumentParser(description="Load raw data for the house price valuation pipeline.")
    parser.add_argument("--raw", type=Path, default=None, help="Path to raw CSV (default: data/raw/AmesHousing.csv)")
    args = parser.parse_args()
    print(args.raw)
    df=load_raw_data(args.raw)
    print(f"Loaded raw data with shape: {df.columns}")


if __name__ == "__main__":
    main()

