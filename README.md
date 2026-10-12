# EngineSense: Predictive Maintenance for Engines

EngineSense predicts how many operating cycles an engine has left before it fails (its **Remaining Useful Life**, or RUL) from its sensor readings, and shows which engines need maintenance first.

Built with Python, Pandas, Scikit-learn and Streamlit, using NASA's C-MAPSS turbofan engine dataset.

## Why this matters

There are three ways to maintain an engine:

1. **Fix it when it breaks.** Cheap until it fails, then very expensive (unplanned downtime).
2. **Service it on a fixed schedule.** Safe, but healthy parts get replaced too early.
3. **Service it when the data says it is wearing out.** This is predictive maintenance, and it is what this project does.

## Result

| Model | Validation RMSE | Test RMSE |
|---|---|---|
| Linear Regression | 17.06 | 16.31 |
| Random Forest | 12.15 | 14.90 |
| **Gradient Boosting** | **11.72** | **13.10** |

RMSE is the typical size of the prediction error, in cycles. The chosen model (Gradient Boosting, retrained on all training engines) has a test RMSE of **13.46 cycles** on 100 engines it never saw during training.

## The data

NASA simulated 100 engines running until failure (file `FD001`). Each row is one engine at one cycle:

- engine number and cycle number
- 3 operating settings
- 21 sensor readings (temperatures, pressures, fan speeds and so on)

There is a separate test set of 100 other engines that stop at some point before failure. NASA provides the true RUL for each of them, which is used to score the model.

## How it works

1. **Label the data** (`features.py`, `add_rul`). Training engines run to failure, so `RUL = last cycle of that engine - current cycle`. RUL is capped at 125: a new engine and a slightly used engine look the same to the sensors, so anything above 125 is treated as "healthy".
2. **Drop useless sensors** (`find_useful_sensors`). 6 of the 21 sensors never change, so they carry no information. 15 are kept.
3. **Build features** (`add_features`). A single reading is noisy. For each sensor the model also gets, over the last 20 cycles: the rolling **mean** (smoothed level), the rolling **standard deviation** (how unstable it is) and the **trend** (how far the level has moved). That gives 60 features.
4. **Split by engine, not by row** (`train.py`). 20 whole engines are held out for validation. Rows from the same engine are nearly identical, so a random row split would leak answers and make the model look better than it is.
5. **Compare three models** and pick the one with the lowest validation RMSE. The test set is only used to report the final score, never to choose.
6. **Dashboard** (`app.py`). Shows the fleet sorted by urgency, each engine's health trend and sensor readings, and the model comparison.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python download_data.py         # only needed if the data/ folder is empty
python train.py                 # takes 2 to 3 minutes
streamlit run app.py
```

## Files

| File | What it does |
|---|---|
| `download_data.py` | Downloads the dataset from NASA |
| `features.py` | Loads the data, labels RUL, builds the rolling features |
| `train.py` | Trains and compares the models, saves the best one |
| `app.py` | Streamlit dashboard |
| `metrics.json` | Model comparison written by `train.py` |
| `model.joblib` | The saved model |

## Limits

- The data is simulated, and `FD001` has one operating condition and one type of fault. Real engines are messier.
- Because of the 125 cap, the model cannot tell a 130-cycle engine from a 300-cycle one. That is acceptable here because the goal is to catch engines that are close to failure.

## Dataset

A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM 2008. Data from the NASA Prognostics Center of Excellence.
