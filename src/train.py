import joblib

from src.data_pipeline import main as dp_main,RANDOM_STATE,build_pipeline,wrap_log_target,MODELS_DIR
import argparse
from pathlib import Path    


import numpy as np 
import pandas as pd 

from src.evaluate import evaluate_model


top_30_features = None
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

def print_top_features(baseline, pipeline_preprocessor, n_top=30) -> None:
    regressor_step = baseline.regressor_.named_steps["regressor"]
    importances = regressor_step.feature_importances_
    feature_names = pipeline_preprocessor.get_feature_names_out()
    
    feature_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
        
    return feature_imp_df.head(n_top)["Feature"].tolist()

def new_val_df(X_val)->pd.DataFrame:
    global top_30_features 
    from src.demoprediction import pred_features
    pred_features_22=pred_features(top_30_features)
    filtered_df = X_val[pred_features_22]
    return filtered_df

def main():
    parser = argparse.ArgumentParser(description="Tune and train the house price model.")
    parser.add_argument("--raw", type=Path, default=None, help="Path to raw CSV (default: data/raw/AmesHousing.csv)")
    parser.add_argument("--model", choices=["xgboost", "lightgbm"], default="xgboost")
    args = parser.parse_args()
    X_train, X_val, y_train_log, y_val_log=dp_main(args.raw) 
    y_val_dollars = np.expm1(y_val_log)

    baseline=fit_final(args.model, BASELINE_PARAMS[args.model], X_train, y_train_log)
    baseline_metrics =evaluate_model(baseline, X_val, y_val_dollars)
    for metric_name, val in baseline_metrics.items():
         print(f"{metric_name.upper():<6} : {val:.4f}")

    print("TOTAL FEATURES ")
    print(len(baseline.regressor_.named_steps["preprocessor"].get_feature_names_out()))


    val_preds_dollars = baseline.predict(X_val.head(5))
    print("\n--- First 5 House Price Predictions ---")
    for i, price in enumerate(val_preds_dollars):
        print(f"House {i+1}: ${price:,.2f}")

    print("\n--- First 5 Actual House Prices ---")
    for i, actual_price in enumerate(y_val_dollars.head(5)):
        print(f"House {i+1}: ${actual_price:,.2f}")

    global top_30_features
    active_preprocessor = baseline.regressor_.named_steps["preprocessor"]
    top_30_features = print_top_features(baseline, active_preprocessor, n_top=30)

    X_val_22=new_val_df(X_val)
    baseline_metrics =evaluate_model(baseline, X_val_22, y_val_dollars)
    for metric_name, val in baseline_metrics.items():
             print(f"{metric_name.upper():<6} : {val:.4f}")


    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"{args.model}_baseline.joblib"
    joblib.dump(baseline, model_path)
    

if __name__ == "__main__":

    main()