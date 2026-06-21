# src/config.py
# ============================================================
# PROJECT CONFIGURATION
# All paths are derived from this file's own location on disk.
# This means paths work correctly regardless of which directory
# Python or Jupyter is launched from — permanently.
# ============================================================

from pathlib import Path

# ----------------------------------------------------------------
# PROJECT ROOT — derived from config.py's location
# config.py lives at:  dfsc/src/config.py
# .parent              → dfsc/src/
# .parent.parent       → dfsc/              ← this is PROJECT_ROOT
# .resolve()           converts to absolute path, removes any ../ ambiguity
# ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- Directory paths ----
RAW_DATA_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DATA_DIR  = PROJECT_ROOT / "data" / "external"
FIGURES_DIR        = PROJECT_ROOT / "outputs" / "figures"
POWERBI_EXPORT_DIR = PROJECT_ROOT / "outputs" / "powerbi"
FORECASTS_DIR      = PROJECT_ROOT / "outputs" / "forecasts"

# ---- File paths ----
TRAIN_FILE = RAW_DATA_DIR / "train.csv"
STORE_FILE = RAW_DATA_DIR / "store.csv"

# ---- Auto-create all output directories on import ----
# This also fixes the "directory does not exist" error that hits
# plt.savefig() later — you never need to manually mkdir again.
for _dir in [
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    FIGURES_DIR / "eda",
    FIGURES_DIR / "forecasts",
    FIGURES_DIR / "inventory",
    POWERBI_EXPORT_DIR,
    FORECASTS_DIR,
    PROJECT_ROOT / "reports",
    PROJECT_ROOT / "excel",
]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---- Convert to strings for pandas/matplotlib compatibility ----
# pathlib Path objects work with open() and pandas read_csv() directly,
# but some older libraries expect plain strings. These str() wrappers
# make both work.
TRAIN_FILE         = str(TRAIN_FILE)
STORE_FILE         = str(STORE_FILE)
FIGURES_DIR        = str(FIGURES_DIR)
POWERBI_EXPORT_DIR = str(POWERBI_EXPORT_DIR)
FORECASTS_DIR      = str(FORECASTS_DIR)
PROCESSED_DATA_DIR = str(PROCESSED_DATA_DIR)
EXTERNAL_DATA_DIR  = str(EXTERNAL_DATA_DIR)

# ================================================================
# ALL OTHER PARAMETERS (unchanged from before)
# ================================================================

# ---- Time split ----
TRAIN_END_DATE       = "2014-12-31"
VALIDATION_START_DATE = "2015-01-01"

# ---- Forecasting ----
FORECAST_HORIZON_DAYS = 90
MIN_HISTORY_DAYS      = 180

# ---- Prophet ----
PROPHET_CHANGEPOINT_PRIOR  = 0.05
PROPHET_SEASONALITY_PRIOR  = 10.0
PROPHET_HOLIDAYS_PRIOR     = 10.0

# ---- XGBoost ----
XGB_N_ESTIMATORS     = 500
XGB_MAX_DEPTH        = 6
XGB_LEARNING_RATE    = 0.05
XGB_SUBSAMPLE        = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

# ---- Lag and rolling ----
LAG_DAYS       = [1, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 28]

# ---- Inventory ----
SERVICE_LEVEL_A      = 0.99
SERVICE_LEVEL_B      = 0.95
SERVICE_LEVEL_C      = 0.90
LEAD_TIME_DAYS       = 7
LEAD_TIME_STD_DAYS   = 2
HOLDING_COST_RATE    = 0.25
ORDERING_COST        = 150
STOCKOUT_PENALTY_RATE = 0.30

# ---- ABC / XYZ thresholds ----
ABC_A_THRESHOLD = 0.80
ABC_B_THRESHOLD = 0.95
XYZ_X_THRESHOLD = 0.20
XYZ_Y_THRESHOLD = 0.50

# config.py
ACCENT   = "#00d4ff"
ACCENT2  = "#ff6b6b"
ACCENT3  = "#51cf66"
ACCENT4  = "#ffd43b"