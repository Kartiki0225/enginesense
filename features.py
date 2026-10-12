"""Loading the engine data and turning raw sensor readings into model features.

Both train.py and app.py import from here, so the model always sees
features that were built in exactly the same way.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# The text files have no header row. Each row is one engine at one cycle.
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ["engine", "cycle"] + SETTING_COLS + SENSOR_COLS

# An engine with 300 cycles left and one with 200 cycles left look the same to
# the sensors: both are healthy. So every RUL above this cap is treated as "healthy".
RUL_CAP = 125

# How many past cycles the rolling features look at.
WINDOW = 20


def load_file(filename: str) -> pd.DataFrame:
    """Read one C-MAPSS text file (values separated by spaces) into a table."""
    return pd.read_csv(DATA_DIR / filename, sep=r"\s+", header=None, names=COLUMNS)


def add_rul(train: pd.DataFrame) -> pd.DataFrame:
    """Add the target column for training data.

    Training engines run until they fail, so the last cycle of each engine is the
    failure point:  RUL = last cycle of that engine - current cycle.
    """
    train = train.copy()
    last_cycle = train.groupby("engine")["cycle"].transform("max")
    train["rul"] = (last_cycle - train["cycle"]).clip(upper=RUL_CAP)
    return train


def find_useful_sensors(train: pd.DataFrame) -> list[str]:
    """Keep only sensors whose reading actually changes.

    Some sensors show the same value on every row. A column that never changes
    cannot help predict anything, so it is dropped.
    """
    return [col for col in SENSOR_COLS if train[col].std() > 0.001]


def add_features(data: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Add rolling-window features for every useful sensor.

    A single reading is noisy. What shows wear is how the readings behave over
    the last few cycles, so for each sensor we add:
      mean  - the smoothed level over the last WINDOW cycles
      std   - how much the reading is shaking over the last WINDOW cycles
      trend - how far the smoothed level has moved compared to WINDOW cycles ago

    Everything is grouped by engine so one engine's history never leaks into another's.
    """
    data = data.sort_values(["engine", "cycle"]).reset_index(drop=True)
    by_engine = data.groupby("engine")

    new_columns = {}
    for col in sensors:
        rolling = by_engine[col].rolling(WINDOW, min_periods=1)
        mean = rolling.mean().reset_index(level=0, drop=True)
        std = rolling.std().reset_index(level=0, drop=True).fillna(0)
        trend = (mean - mean.groupby(data["engine"]).shift(WINDOW)).fillna(0)

        new_columns[f"{col}_mean"] = mean
        new_columns[f"{col}_std"] = std
        new_columns[f"{col}_trend"] = trend

    return pd.concat([data, pd.DataFrame(new_columns)], axis=1)


def feature_names(sensors: list[str]) -> list[str]:
    """The exact list of columns that go into the model, in a fixed order."""
    names = list(sensors)
    for col in sensors:
        names += [f"{col}_mean", f"{col}_std", f"{col}_trend"]
    return names
