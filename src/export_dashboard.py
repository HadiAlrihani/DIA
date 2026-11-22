import pickle
import pandas as pd

TEST_START = "2025-01-01"
INFO = ["days_since_failure", "well_mtbf", "failures_prior_365d", "last_failed_component"]

# 1. past weeks: risk cores the model gave on test mondays
scores = pd.read_pickle("outputs/test_scores.pkl")
mt = pd.read_pickle("outputs/model_table.pkl")
weekly = scores[scores.date.dt.dayofweek == 0][["uwi", "date", "risk", "y"]]
weekly = weekly.merge(mt[["uwi", "date", "field_operating_area"] + INFO], on=["uwi", "date"], how="left")

# 2. lastet day: score every running well with the trained model
bundle = pickle.load(open("outputs/model.pkl", "rb"))
feat = pd.read_pickle("outputs/features.pkl")
train_rows = mt[(mt.date >= "2021-04-01") & (mt.date < "2024-07-01")]
for c in ["field_operating_area", "pcp_pump_make", "pcp_pump_model", "last_failed_component"]:
    keep = train_rows[c].astype(str).value_counts()
    keep = keep[keep >= 2000].index
    feat[c] = feat[c].astype(str).where(feat[c].astype(str).isin(keep), "OTHER").astype("category")
latest_day = feat.date.max()
today = feat[(feat.date == latest_day) & feat.has_pcp & (feat.hours_on_prod > 0)].copy()
today["risk"] = bundle["model"].predict_proba(today[bundle["features"]])[:, 1]
today["y"] = float("nan")
latest = today[["uwi", "date", "risk", "y", "field_operating_area"] + INFO]

weekly = pd.concat([weekly, latest], ignore_index=True)
weekly["field_operating_area"] = weekly.field_operating_area.astype(str)
weekly["last_failed_component"] = weekly.last_failed_component.astype(str)
weekly["rank"] = weekly.groupby("date").risk.rank(ascending=False, method="first").astype(int)
weekly["risk"] = weekly.risk.round(4)
weekly.to_csv("outputs/dash_weekly.csv", index=False)

# 3. well locations (median of the daily coordinates)
cols = ["uwi", "field_operating_area", "surface_latitude", "surface_longitude"]
loc = pd.read_csv("data/daily_production.csv", usecols=cols, dtype={"uwi": str})
wells = (loc[loc.uwi.isin(weekly.uwi)].dropna().groupby("uwi").agg(field=("field_operating_area", "first"),lat=("surface_latitude", "median"),lon=("surface_longitude", "median")).reset_index())
wells.to_csv("outputs/dash_wells.csv", index=False)

# 4. failures since the previous test started
ev = pd.read_csv("outputs/failure_events.csv", parse_dates=["date"], dtype={"uwi": str})
ev[ev.date >= TEST_START][["uwi", "date", "component"]].to_csv("outputs/dash_failures.csv", index=False)

# 5. check
past = weekly[weekly.y.notna()]
print(f"Weeks: {past.date.nunique()} test Mondays + latest day {latest_day:%Y-%m-%d}")
print(f"Rows in dash_weekly.csv: {len(weekly):,}  (wells per week ~{weekly.groupby('date').size().mean():.0f})")
print(f"Wells with map coordinates: {len(wells):,} of {weekly.uwi.nunique():,}")
print(f"Top-50 hit rate over test weeks: {past[past['rank'] <= 50].y.mean():.1%}")
print(f"Latest top 5 fields: {latest.sort_values('risk', ascending=False).head(50).field_operating_area.astype(str).value_counts().head(5).to_dict()}")