import numpy as np
import pandas as pd

df = pd.read_pickle("outputs/daily_clean.pkl")
events = pd.read_csv("outputs/failure_events.csv", parse_dates=["date"])
df["uwi"] = df.uwi.astype(str)
events["uwi"] = events.uwi.astype(str)
print(f"Loaded {len(df):,} well-days, {len(events):,} failures")

# 1. rolling sensor feature
SENSORS = ["pcp_pump_rod_torque_ftlb", "pcp_pump_electric_current_a", "pump_efficiency_pct","pcp_pump_rpm", "pcp_pump_volume_throughput_m3d", "total_fluid_m3","oil_produced_m3", "water_cut", "casing_pressure_kpa", "tubing_pressure_kpa","torque_per_rpm"]
KEY = ["pcp_pump_rod_torque_ftlb", "pcp_pump_electric_current_a", "pump_efficiency_pct","total_fluid_m3", "torque_per_rpm"]           # get extra "vs normal" features

g = df.groupby("uwi", sort=False)

def roll(col, window, how="mean"):
    r = g[col].rolling(window, min_periods=max(2, window // 3))
    return getattr(r, how)().reset_index(level=0, drop=True)

feat = {}
for c in SENSORS:
    m7, m30, m90 = roll(c, 7), roll(c, 30), roll(c, 90)
    feat[f"{c}_m7"] = m7                                       # last week's level
    feat[f"{c}_m30"] = m30                                     # last month's level
    feat[f"{c}_r7_90"] = m7 / m90.replace(0, np.nan)           # this week vs last 3 months
    feat[f"{c}_sd7"] = roll(c, 7, "std") / m30.abs().replace(0, np.nan)
    if c in KEY:
        sd90 = roll(c, 90, "std")
        feat[f"{c}_z7_90"] = (m7 - m90) / sd90.replace(0, np.nan)   # unusual for THIS well?
        feat[f"{c}_chg14"] = m7 - m7.groupby(df.uwi).shift(14)       # 2-week trend

# 2. Recent downtime
for c, w in [("hours_down", 7), ("hours_down", 30), ("lift_downtime_hrs", 30),("lift_downtime_hrs", 90), ("torqued_flag", 30)]:
    feat[f"{c}_sum{w}"] = g[c].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)

# How often were meanual readings updated
for c in ["pcp_pump_rod_torque_ftlb", "pcp_pump_electric_current_a", "pcp_pump_rpm"]:
    changed = (g[c].diff().fillna(0) != 0).astype("float32")
    feat[f"{c}_changes30"] = changed.groupby(df.uwi).rolling(30, min_periods=1).sum().reset_index(level=0, drop=True)

X = pd.DataFrame(feat, index=df.index).astype("float32")

# 3. well / pump facts
X["rated_throughput"] = df.pcp_pump_displacement_m3d100rpm * df.pcp_pump_rpm / 100
X["pump_displacement"] = df.pcp_pump_displacement_m3d100rpm
X["well_age_yrs"] = (df.production_date - df.on_production_date).dt.days / 365.25
X["month"] = df.production_date.dt.month
for c in ["field_operating_area", "pcp_pump_make", "pcp_pump_model"]:
    X[c] = df[c]

# 4. failure history (ONLY FAILURES ON OR BEFORE TODAY)
ev = events.sort_values(["uwi", "date"]).copy()
ev["run_life"] = ev.groupby("uwi").date.diff().dt.days            # days this pump lasted
ev["mtbf"] = ev.groupby("uwi").run_life.transform(lambda s: s.expanding().mean())
ev["n_failures"] = ev.groupby("uwi").cumcount() + 1

key = df[["uwi", "production_date"]].reset_index().sort_values("production_date")
last = pd.merge_asof(key, ev.sort_values("date"), left_on="production_date", right_on="date",by="uwi", direction="backward").set_index("index").sort_index()

X["days_since_failure"] = (df.production_date - last.date).dt.days
X["last_failed_component"] = last.component.fillna("none").astype("category")
X["last_run_life"] = last.run_life
X["well_mtbf"] = last.mtbf
X["run_life_vs_mtbf"] = X.days_since_failure / X.well_mtbf.replace(0, np.nan)
X["n_failures_to_date"] = last.n_failures.fillna(0)
X["last_repair_cost"] = last.repair_cost

cnt = np.zeros(len(df), dtype="float32")
for uwi, idx in df.groupby("uwi", sort=False).indices.items():
    d = ev.date[ev.uwi == uwi].values
    if len(d):
        t = df.production_date.values[idx]
        cnt[idx] = (np.searchsorted(d, t, side="right") - np.searchsorted(d, t - np.timedelta64(365, "D"), side="right"))
X["failures_prior_365d"] = cnt

# 5. keep the ID columns and save it
X["uwi"] = df.uwi.values
X["date"] = df.production_date.values
X["hours_on_prod"] = df.hours_on_prod.values
X["oil_m3"] = df.oil_produced_m3.values
X["has_pcp"] = df.groupby("uwi").pcp_pump_rpm.transform(lambda s: s.notna().any()).values
X.to_pickle("outputs/features.pkl")

n_feat = X.shape[1] - 5
print(f"Built {n_feat} features for {len(X):,} well-days")
print("\nExample: a well with torque data, its last 5 days")
cols = ["date", "pcp_pump_rod_torque_ftlb_m7", "pcp_pump_rod_torque_ftlb_z7_90","days_since_failure", "well_mtbf", "failures_prior_365d"]
recent = X[(X.date == X.date.max()) & (X.pcp_pump_rod_torque_ftlb_m7 > 0) & (X.failures_prior_365d >= 2)]
w = recent.sort_values("uwi").uwi.iloc[0]
print(X.loc[X.uwi == w, cols].tail(5).to_string(index=False, float_format="%.2f"))
print(f"\nWell-days where RPM changed at least once in the last 30 days: "f"{(X.pcp_pump_rpm_changes30 > 0).mean():.0%}")
print(f"Well-days with no earlier failure (days_since_failure missing): "f"{X.days_since_failure.isna().mean():.0%}")