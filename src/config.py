"""Central configuration for the FRD (arecanut fruit rot) modelling pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "FRD data-revised.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "frd_clean.csv"
CLEANING_LOG_CSV = PROCESSED_DIR / "cleaning_log.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "docs" / "figures"
MODEL_DIR = ROOT / "models"

SEED = 42

# Canonical column names (raw sheet label -> short name)
PARAM_MAP = {
    "Average Tmax": "tmax",
    "Average Tmin": "tmin",
    "Average RH1": "rh1",
    "Average RH2": "rh2",
    "Average Max. Wind speed": "wind",
    "Total rainfall": "rain",
    "No. of rainy days": "rainy_days",
    "CFNR": "cfnr",
    "Disease Severity": "severity",
}
WEATHER = ["tmax", "tmin", "rh1", "rh2", "wind", "rain", "rainy_days"]
TARGETS = ["cfnr", "severity"]
REGIONS = ["Coastal", "Maidan", "Malnad"]
N_WEEKS = 18

# --- Data-cleaning rules (documented in docs/RESULTS.md) -------------------
# Rule 1: physically implausible values for the study area -> missing.
PHYSICAL_BOUNDS = {
    "tmax": (15.0, 45.0),
    "tmin": (10.0, 35.0),
    "rh1": (0.0, 100.0),
    "rh2": (0.0, 100.0),
    "wind": (0.0, 60.0),
    "rain": (0.0, 2000.0),
    "rainy_days": (0, 7),
}
# Rule 2: spatio-temporal spike test for highly skewed variables.
SPIKE_VARS = ["wind", "rain"]
# A value is flagged when it exceeds BOTH adjacent weeks AND the regional
# same-week median by this factor. Thresholds sit at the natural break in the
# ranked min-ratio distribution (wind: 3.91 vs next 3.26; rain: 7.80 vs next
# 5.21, the latter part of a spatially corroborated 3-station storm in W14-2019).
SPIKE_RATIO = {"wind": 3.5, "rain": 6.0}
SPIKE_MIN_ABS = {"wind": 5.0, "rain": 100.0}  # ignore small absolute values

# --- Risk bands on predicted disease severity (%) --------------------------
# Default, data-informed thresholds; confirm against the domain scale used
# by the plant-pathology team before publication.
RISK_BANDS = [
    (2.0, "Nil"),
    (15.0, "Low"),
    (35.0, "Moderate"),
    (float("inf"), "High"),
]


def severity_to_band(sev: float) -> str:
    for upper, name in RISK_BANDS:
        if sev < upper:
            return name
    return RISK_BANDS[-1][1]
