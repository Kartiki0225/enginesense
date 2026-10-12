"""EngineSense dashboard: which engines need maintenance soon?

Run:  streamlit run app.py      (train the model first with: python train.py)
"""

import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from features import add_features, load_file

HERE = Path(__file__).parent

st.set_page_config(page_title="EngineSense", page_icon="⚙️", layout="wide")


@st.cache_resource
def load_model() -> dict:
    return joblib.load(HERE / "model.joblib")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    """Predict RUL for every cycle of every test engine (engines the model never saw)."""
    bundle = load_model()
    data = add_features(load_file("test_FD001.txt"), bundle["sensors"])
    data["predicted_rul"] = bundle["model"].predict(data[bundle["features"]]).clip(min=0)
    return data


@st.cache_data
def load_actual_rul() -> pd.Series:
    """NASA's answer file: the true RUL of each test engine at its last recorded cycle."""
    actual = pd.read_csv(HERE / "data" / "RUL_FD001.txt", header=None)[0]
    actual.index = actual.index + 1  # engine numbers start at 1
    return actual


def status_of(rul: float, limit: int) -> str:
    if rul < limit:
        return "Critical"
    if rul < 2 * limit:
        return "Warning"
    return "Healthy"


if not (HERE / "model.joblib").exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

bundle = load_model()
data = load_predictions()
actual_rul = load_actual_rul()
metrics = json.loads((HERE / "metrics.json").read_text())

# ---------------------------------------------------------------- sidebar

st.sidebar.title("EngineSense")
st.sidebar.caption("Predictive maintenance for engines")
limit = st.sidebar.slider(
    "Flag engines with fewer than this many cycles left",
    min_value=10, max_value=60, value=30, step=5,
)
st.sidebar.markdown(
    f"**Critical:** under {limit} cycles  \n"
    f"**Warning:** under {2 * limit} cycles  \n"
    f"**Healthy:** {2 * limit} cycles or more"
)

# One row per engine: its latest cycle and the prediction made there.
fleet = data.groupby("engine").tail(1)[["engine", "cycle", "predicted_rul"]].copy()
fleet["status"] = fleet["predicted_rul"].apply(status_of, limit=limit)
fleet = fleet.sort_values("predicted_rul").reset_index(drop=True)

fleet_tab, engine_tab, model_tab = st.tabs(["Fleet overview", "Engine detail", "Model"])

# ---------------------------------------------------------------- fleet overview

with fleet_tab:
    st.header("Fleet overview")
    st.caption("Each engine's Remaining Useful Life (RUL), predicted from its latest sensor readings.")

    counts = fleet["status"].value_counts()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Engines monitored", len(fleet))
    col2.metric("Critical", int(counts.get("Critical", 0)))
    col3.metric("Warning", int(counts.get("Warning", 0)))
    col4.metric("Healthy", int(counts.get("Healthy", 0)))

    st.subheader("Maintenance priority list")
    st.dataframe(
        fleet,
        hide_index=True,
        width="stretch",
        column_config={
            "engine": "Engine",
            "cycle": "Cycles run",
            "predicted_rul": st.column_config.ProgressColumn(
                "Predicted cycles left", format="%.0f", min_value=0, max_value=125
            ),
            "status": "Status",
        },
    )

# ---------------------------------------------------------------- engine detail

with engine_tab:
    st.header("Engine detail")

    engine_id = st.selectbox(
        "Engine (most urgent first)",
        fleet["engine"],
        format_func=lambda e: f"Engine {e}",
    )
    history = data[data["engine"] == engine_id]
    latest = fleet[fleet["engine"] == engine_id].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", latest["status"])
    col2.metric("Predicted cycles left", f"{latest['predicted_rul']:.0f}")
    col3.metric("Actual cycles left", int(actual_rul[engine_id]), help="From NASA's answer file")
    col4.metric("Cycles run so far", int(latest["cycle"]))

    st.subheader("Health trend")
    st.caption("Predicted cycles left at every point in this engine's life. A falling line means wear.")
    st.line_chart(history, x="cycle", y="predicted_rul", x_label="Cycle", y_label="Predicted cycles left")

    st.subheader("Sensor readings")
    sensor = st.selectbox("Sensor", bundle["sensors"], index=bundle["sensors"].index("sensor_11"))
    readings = history[["cycle", sensor, f"{sensor}_mean"]].rename(
        columns={sensor: "Raw reading", f"{sensor}_mean": "Rolling average"}
    )
    # Sensor values move in a narrow band (for example 47 to 48), so the y-axis
    # must not start at zero or the drift would look like a flat line.
    chart = (
        alt.Chart(readings.melt("cycle", var_name="Series", value_name="value"))
        .mark_line()
        .encode(
            x=alt.X("cycle", title="Cycle"),
            y=alt.Y("value", title=sensor, scale=alt.Scale(zero=False)),
            color=alt.Color("Series", title=None),
        )
    )
    st.altair_chart(chart)

# ---------------------------------------------------------------- model

with model_tab:
    st.header("Model")
    st.caption(
        "RMSE is the typical size of the error, in cycles. Lower is better. "
        "Validation = 20 training engines held out. Test = NASA's 100 separate test engines."
    )

    comparison = pd.DataFrame(metrics["comparison"]).rename(
        columns={"model": "Model", "validation_rmse": "Validation RMSE", "test_rmse": "Test RMSE"}
    )
    st.dataframe(comparison, hide_index=True)
    st.markdown(
        f"**Model in use:** {metrics['best_model']}, retrained on all 100 training engines. "
        f"Test RMSE **{metrics['final_test_rmse']} cycles**."
    )

    st.subheader("Predicted vs actual on the test engines")
    check = fleet.assign(actual_rul=fleet["engine"].map(actual_rul))
    st.scatter_chart(
        check, x="actual_rul", y="predicted_rul",
        x_label="Actual cycles left", y_label="Predicted cycles left",
    )
    st.caption("Points near the diagonal are good predictions. Training labels are capped at 125 cycles, "
               "so the model never predicts more than about 125.")

    model = bundle["model"]
    if hasattr(model, "feature_importances_"):
        st.subheader("Features the model relies on most")
        importance = (
            pd.Series(model.feature_importances_, index=bundle["features"])
            .sort_values(ascending=False)
            .head(10)
        )
        st.bar_chart(importance, horizontal=True, x_label="Importance", y_label="Feature")
