from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np



def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs(y_true - y_pred) / y_true)),
        "rmsle": float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))),
    }

def evaluate_model(baseline, X_val, y_val_dollars):

    return regression_metrics(y_val_dollars, baseline.predict(X_val))

def main() -> None:
    pass
if __name__ == "__main__":
    main()