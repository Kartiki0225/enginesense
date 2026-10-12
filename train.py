"""Train three models to predict Remaining Useful Life (RUL) and keep the best one.

Run:  python train.py
Output: model.joblib (the best model) and metrics.json (the comparison table).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import add_features, add_rul, feature_names, find_useful_sensors, load_file

HERE = Path(__file__).parent
SEED = 42


def build_models() -> dict:
    return {
        # Simple baseline. Scaling puts every sensor on the same range first.
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        # Many decision trees, each trained on a random sample, then averaged.
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED
        ),
        # Trees built one after another, each fixing the errors of the previous ones.
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=SEED
        ),
    }


def load_test_set(sensors: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """The test engines stop before failure. NASA gives the true RUL at their last cycle."""
    test = add_features(load_file("test_FD001.txt"), sensors)
    last_rows = test.groupby("engine").tail(1).sort_values("engine")
    true_rul = pd.read_csv(HERE / "data" / "RUL_FD001.txt", header=None)[0]
    return last_rows, true_rul


def main() -> None:
    # 1. Load the training engines and build features.
    train = add_rul(load_file("train_FD001.txt"))
    sensors = find_useful_sensors(train)
    features = feature_names(sensors)
    train = add_features(train, sensors)
    print(f"{train['engine'].nunique()} engines, {len(train)} rows")
    print(f"{len(sensors)} useful sensors out of 21, {len(features)} features")

    # 2. Hold out 20 whole engines for validation.
    #    Splitting by engine (not by random rows) matters: rows of one engine are
    #    almost identical to each other, so mixing them would let the model cheat.
    engines = train["engine"].unique()
    rng = np.random.default_rng(SEED)
    val_engines = rng.choice(engines, size=20, replace=False)
    is_val = train["engine"].isin(val_engines)
    fit_part, val_part = train[~is_val], train[is_val]

    test_rows, test_rul = load_test_set(sensors)

    # 3. Train each model and measure its error in cycles (RMSE, lower is better).
    results = []
    for name, model in build_models().items():
        model.fit(fit_part[features], fit_part["rul"])
        val_rmse = root_mean_squared_error(val_part["rul"], model.predict(val_part[features]))
        test_rmse = root_mean_squared_error(test_rul, model.predict(test_rows[features]))
        results.append(
            {"model": name, "validation_rmse": round(val_rmse, 2), "test_rmse": round(test_rmse, 2)}
        )
        print(f"{name:18s}  validation RMSE {val_rmse:6.2f}   test RMSE {test_rmse:6.2f}")

    # 4. Pick the winner using validation only. The test set is never used to choose.
    best = min(results, key=lambda row: row["validation_rmse"])
    print("Best model:", best["model"])

    # 5. Retrain the winner on all 100 training engines and save it.
    final_model = build_models()[best["model"]]
    final_model.fit(train[features], train["rul"])
    final_rmse = root_mean_squared_error(test_rul, final_model.predict(test_rows[features]))
    print(f"Final model test RMSE {final_rmse:.2f} cycles")

    joblib.dump({"model": final_model, "sensors": sensors, "features": features}, HERE / "model.joblib")
    metrics = {"best_model": best["model"], "final_test_rmse": round(final_rmse, 2), "comparison": results}
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved model.joblib and metrics.json")


if __name__ == "__main__":
    main()
