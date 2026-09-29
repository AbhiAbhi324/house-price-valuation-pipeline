from src.data_pipeline import main as dp_main,RANDOM_STATE,build_pipeline,wrap_log_target
import argparse
from pathlib import Path    


import numpy as np 
import pandas as pd 



BASELINE_PARAMS = {
    "xgboost": dict(n_estimators=1000, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.6),
    "lightgbm": dict(n_estimators=1000, learning_rate=0.05, num_leaves=15, subsample=0.8, colsample_bytree=0.6),
}


def make_regressor(name: str, params: dict):
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)

    elif name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    raise ValueError(f"Unknown model name: {name}")


def fit_final(model_name: str, params: dict, X: pd.DataFrame, y_log: pd.Series):
    model = wrap_log_target(build_pipeline(make_regressor(model_name, params)))
    model.fit(X, np.expm1(y_log))
    return model


def main():
    parser = argparse.ArgumentParser(description="Tune and train the house price model.")
    parser.add_argument("--raw", type=Path, default=None, help="Path to raw CSV (default: data/raw/AmesHousing.csv)")
    parser.add_argument("--model", choices=["xgboost", "lightgbm"], default="xgboost")
    args = parser.parse_args()
    X_train, X_val, y_train_log, y_val_log=dp_main(args.raw) 
    y_val_dollars = np.expm1(y_val_log)

    baseline=fit_final(args.model, BASELINE_PARAMS[args.model], X_train, y_train_log)
    print(baseline.predict(X_val.iloc[0:1]))
    print(y_val_dollars.iloc[0])

if __name__ == "__main__":

    main()