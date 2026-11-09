"""Feature engineering shared by training and the Streamlit app.

Inputs available at prediction time (per the study design):
  region, week number (1-18), one week's weather (7 variables),
  and OPTIONALLY the previous week's CFNR and disease severity.
"""
import numpy as np
import pandas as pd

import config as C

BASE_FEATURES = (["week"] + [f"region_{r}" for r in C.REGIONS] + C.WEATHER +
                 ["trange", "tmean", "rh_mean", "rh_diff", "rain_intensity"])
HISTORY_FEATURES = ["prev_cfnr", "prev_severity"]


def feature_list(with_history: bool):
    return BASE_FEATURES + (HISTORY_FEATURES if with_history else [])


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for r in C.REGIONS:
        out[f"region_{r}"] = (out["region"] == r).astype(int)
    out["trange"] = out["tmax"] - out["tmin"]
    out["tmean"] = (out["tmax"] + out["tmin"]) / 2
    out["rh_mean"] = (out["rh1"] + out["rh2"]) / 2
    out["rh_diff"] = out["rh1"] - out["rh2"]
    out["rain_intensity"] = out["rain"] / np.maximum(out["rainy_days"], 1)
    return out


def build_training_frame(clean: pd.DataFrame) -> pd.DataFrame:
    """Add derived features and the (true) previous-week targets per series."""
    df = add_derived(clean.sort_values(["series_id", "week"]))
    g = df.groupby("series_id")
    # Week 1 has no previous week; disease is absent before the season starts.
    df["prev_cfnr"] = g["cfnr"].shift(1).fillna(0.0)
    df["prev_severity"] = g["severity"].shift(1).fillna(0.0)
    return df.reset_index(drop=True)


def build_inference_row(region: str, week: int, weather: dict,
                        prev_cfnr=None, prev_severity=None) -> pd.DataFrame:
    row = dict(region=region, week=int(week), **{k: float(weather[k]) for k in C.WEATHER})
    if prev_cfnr is not None and prev_severity is not None:
        row["prev_cfnr"] = float(prev_cfnr)
        row["prev_severity"] = float(prev_severity)
    return add_derived(pd.DataFrame([row]))
