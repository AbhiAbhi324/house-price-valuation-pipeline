from pathlib import Path    

import pandas as pd 
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler



ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

RAW_FILE_NAME ="AmesHousing.csv"

RANDOM_STATE = 42
TARGET = "saleprice"

SPARSE_COLS = ["pool_qc", "misc_feature", "alley", "garage_yr_blt"]
REDUNDANT_COLS = [
    "1st_flr_sf", "2nd_flr_sf", "total_bsmt_sf",
    "full_bath", "half_bath", "bsmt_full_bath", "bsmt_half_bath",
    "year_built", "garage_area",
]
DROP_COLS = SPARSE_COLS + REDUNDANT_COLS


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
def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['order', 'pid'], errors='ignore')
    df = df[df['gr_liv_area'] <= 4000]
    if "garage_yr_blt" in df.columns:
        df['garage_yr_blt'] = df['garage_yr_blt'].replace(2207, 2007)
    return df

def make_splits(df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    X = df.drop(columns=[TARGET])
    y = np.log1p(df[TARGET]).rename(TARGET)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, val_index in splitter.split(df, df['overall_qual']):
       X_train = X.iloc[train_index]
       X_val = X.iloc[val_index]
       y_train = y.iloc[train_index]
       y_val = y.iloc[val_index]
    return X_train, X_val, y_train, y_val
def save_processed_data(X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame)->None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(OUT_DIR / "X_train.csv", index=False)
    X_val.to_csv(OUT_DIR / "X_val.csv", index=False)
    y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
    y_val.to_csv(OUT_DIR / "y_val.csv", index=False)
    print (f"Saved processed data to {OUT_DIR}")

class FeatureEngineer(BaseEstimator, TransformerMixin):

    required_features = ["1st_flr_sf", "2nd_flr_sf", "total_bsmt_sf", "full_bath", "half_bath",
                "bsmt_full_bath", "bsmt_half_bath", "yr_sold", "year_built"]

    def fit(self, X, y=None):
        X = self._check_frame(X)
        missing = [c for c in self.required_features if c not in X.columns]
        if missing:
            raise ValueError(
                f"Missing raw columns {missing}. FeatureEngineer expects PRE-engineering data - "
                "regenerate partitions with `python -m src.data_pipeline`."
            )
        self.raw_columns_ = list(X.columns)
        self.numeric_columns_ = X.select_dtypes(include="number").columns.tolist()
        self.categorical_columns_ = [c for c in X.columns if c not in self.numeric_columns_]
        return self

    def transform(self, X):
        X = self._check_frame(X).reindex(columns=self.raw_columns_).copy()
        for col in self.numeric_columns_:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
        for col in self.categorical_columns_:
            X[col] = X[col].astype(object).where(X[col].notna(), np.nan)

        X["total_sq_ft"] = X["1st_flr_sf"] + X["2nd_flr_sf"] + X["total_bsmt_sf"]
        X["total_bathrooms"] = (
            X["full_bath"] + 0.5 * X["half_bath"] + X["bsmt_full_bath"] + 0.5 * X["bsmt_half_bath"]
        )
        X["property_age"] = X["yr_sold"] - X["year_built"]
        return X.drop(columns=DROP_COLS, errors="ignore")

    @staticmethod
    def _check_frame(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame with named columns.")
        return X

def build_pipeline(regressor)-> Pipeline:
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocessor", build_preprocessor()),
        ("regressor", regressor),
    ])
def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num, make_column_selector(dtype_include=np.number)),
        ("cat", cat, make_column_selector(dtype_exclude=np.number)),
    ])
    
def wrap_log_target(pipeline)-> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=pipeline, func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )



def main(Path):
    
    df=clean_raw_data(load_raw_data(Path))
    print(f"Loaded raw data with shape: {df.shape}")

    splits=make_splits(df)
    save_processed_data(*splits)
    return splits


