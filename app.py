"""FRD predictor - Streamlit web app.

Run:  .venv\\Scripts\\streamlit run app.py
Predicts same-week CFNR (fallen nuts/tree) and disease severity (%) of arecanut
fruit rot from one week of weather + region + week, optionally using last
week's observed CFNR and severity.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402
import features as F  # noqa: E402

LABELS = {"tmax": ("Max temperature", "°C"), "tmin": ("Min temperature", "°C"),
          "rh1": ("Morning RH", "%"), "rh2": ("Evening RH", "%"),
          "wind": ("Max wind speed", ""), "rain": ("Total rainfall", "mm"),
          "rainy_days": ("Rainy days", "days")}
HARD_LIMITS = {"tmax": (10.0, 45.0), "tmin": (5.0, 35.0), "rh1": (0.0, 100.0),
               "rh2": (0.0, 100.0), "wind": (0.0, 60.0), "rain": (0.0, 2000.0),
               "rainy_days": (0, 7)}
BAND_STYLE = {  # status palette: always icon + label, never colour alone
    "Nil": ("#0ca30c", "✅", "No disease expected"),
    "Low": ("#fab219", "🟡", "Monitor the garden"),
    "Moderate": ("#ec835a", "🟠", "Consider protective measures"),
    "High": ("#d03b3b", "🔴", "High disease pressure — act promptly"),
}


@st.cache_resource
def load_bundle():
    for name in ("frd_models.joblib", "frd_models_quick.joblib"):
        p = ROOT / "models" / name
        if p.exists():
            return joblib.load(p), name
    return None, None


def defaults_for(bundle, region, week):
    if "defaults" in bundle:
        return bundle["defaults"][region][int(week)]
    clean = ROOT / "data/processed/frd_clean.csv"
    df = pd.read_csv(clean)
    return df[(df.region == region) & (df.week == week)][C.WEATHER].median().to_dict()


def predict(bundle, region, week, weather, prev_cfnr=None, prev_sev=None):
    hist = prev_cfnr is not None and prev_sev is not None
    variant = "with_history" if hist else "weather_only"
    row = F.build_inference_row(region, week, weather, prev_cfnr, prev_sev)
    out = {"variant": variant}
    for target in C.TARGETS:
        m = bundle["models"][(variant, target)]
        p = float(m["estimator"].predict(row[m["features"]])[0])
        hi_cap = 100.0 if target == "severity" else None
        p = float(np.clip(p, 0, hi_cap))
        hw = m["pi_halfwidth"]
        if isinstance(hw, dict):  # region-conditional (Mondrian) conformal interval
            hw = hw[region]
        out[target] = dict(pred=p, lo=max(0.0, p - hw),
                           hi=min(hi_cap, p + hw) if hi_cap else p + hw,
                           cv_rmse=m["cv_rmse"], cv_r2=m["cv_r2"])
    out["band"] = C.severity_to_band(out["severity"]["pred"])
    return out


def range_warnings(bundle, region, weather):
    rng = bundle["input_ranges"][region]
    msgs = []
    for v, x in weather.items():
        lo, hi = rng[v]
        if x < lo or x > hi:
            msgs.append(f"{LABELS[v][0]} = {x:g} is outside the range seen in {region} "
                        f"training data ({lo:g}–{hi:g}); the prediction is an extrapolation.")
    return msgs


# ---------------------------------------------------------------- page --
st.set_page_config(page_title="Arecanut Fruit Rot Predictor", page_icon="🌴", layout="wide")
bundle, bundle_name = load_bundle()
st.title("🌴 Arecanut Fruit Rot Predictor")
st.caption("Estimates this week's **fallen nuts per tree (CFNR)** and **disease severity (%)** "
           "from one week of weather.")

if bundle is None:
    st.error("No trained model found in `models/`. Run the training pipeline first.")
    st.stop()
if bundle.get("quick"):
    st.warning("⚠️ Using the **preliminary** model (smoke-test settings). Results will "
               "improve when the final model (`models/frd_models.joblib`) is trained.")

(tab_single,) = st.tabs(["Single prediction"])

with tab_single:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("1 · Where and when")
        c1, c2 = st.columns(2)
        region = c1.selectbox("Region", C.REGIONS, index=C.REGIONS.index("Malnad"))
        week = c2.number_input("Week of season (W1–W18)", 1, C.N_WEEKS, 10, step=1)
        d = defaults_for(bundle, region, week)

        st.subheader("2 · This week's weather")
        st.caption(f"Pre-filled with the typical (median) values for {region}, week {week}. "
                   "Replace them with your observations.")
        weather = {}
        cols = st.columns(2)
        for i, v in enumerate(C.WEATHER):
            name, unit = LABELS[v]
            lo, hi = HARD_LIMITS[v]
            label = f"{name} ({unit})" if unit else name
            if v == "rainy_days":
                weather[v] = cols[i % 2].number_input(label, int(lo), int(hi),
                                                      int(round(d[v])), step=1,
                                                      key=f"{v}_{region}_{week}")
            else:
                weather[v] = cols[i % 2].number_input(label, float(lo), float(hi),
                                                      float(round(d[v], 1)), step=0.1,
                                                      format="%.1f", key=f"{v}_{region}_{week}")

        st.subheader("3 · Last week's disease (optional)")
        use_hist = st.toggle("I know last week's CFNR and severity",
                             help="Supplying last week's values makes the prediction "
                                  "considerably more accurate.")
        prev_cfnr = prev_sev = None
        if use_hist:
            h1, h2 = st.columns(2)
            prev_cfnr = h1.number_input("Last week's CFNR (nuts/tree)", 0.0, 500.0, 0.0, step=1.0)
            prev_sev = h2.number_input("Last week's severity (%)", 0.0, 100.0, 0.0, step=0.5)
            if week == 1:
                st.caption("Week 1 is the start of the season — last week's values are normally 0.")

    with right:
        st.subheader("Prediction")
        res = predict(bundle, region, week, weather, prev_cfnr, prev_sev)
        color, icon, advice = BAND_STYLE[res["band"]]
        st.markdown(
            f"<div style='border-left:6px solid {color};padding:12px 16px;border-radius:6px;"
            f"background:rgba(127,127,127,0.08);margin-bottom:12px'>"
            f"<div style='font-size:0.85rem;opacity:0.75'>Risk band</div>"
            f"<div style='font-size:1.6rem;font-weight:600'>{icon} {res['band']}</div>"
            f"<div style='opacity:0.85'>{advice}</div></div>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        s, c = res["severity"], res["cfnr"]
        m1.metric("Disease severity", f"{s['pred']:.1f} %")
        m1.caption(f"{int(bundle['pi_level'] * 100)}% interval: {s['lo']:.1f} – {s['hi']:.1f} %")
        m2.metric("Fallen nuts (CFNR)", f"{c['pred']:.1f} per tree")
        m2.caption(f"{int(bundle['pi_level'] * 100)}% interval: {c['lo']:.1f} – {c['hi']:.1f}")

        mode = ("weather + last week's disease" if res["variant"] == "with_history"
                else "weather only")
        st.info(f"Model used: **{bundle['model_name']}**, {mode}. On locations it had never "
                f"seen, typical error was ±{s['cv_rmse']:.1f} % severity and "
                f"±{c['cv_rmse']:.1f} nuts/tree (RMSE).")
        for w in range_warnings(bundle, region, weather):
            st.warning(w)
        bands = " · ".join(
            f"{BAND_STYLE[b][1]} {b}: "
            + (f"< {hi:g} %" if i == 0 else
               (f"≥ {C.RISK_BANDS[i - 1][0]:g} %" if hi == float("inf")
                else f"{C.RISK_BANDS[i - 1][0]:g}–{hi:g} %"))
            for i, (hi, b) in enumerate(C.RISK_BANDS))
        st.caption(f"Risk bands are based on predicted severity — {bands}.")
