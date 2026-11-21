import re
import pandas as pd

#Job types that say something was broken and we fixed it
FAIL_JOB = re.compile(r"Rod Repair|Rod Break|Polish Rod|Pump Repair|PCP Replacement|Tubing Repair|"r"Rotor Change|BHP Replacement|ESP Replacement|Packer Repair", re.I)

MERGE_DAYS = 7 # events closer than this on one well = same failure

def component(text):
    t = str(text).lower()
    if "rod" in t or "rotor" in t: return "Rods"
    if "pump" in t or "pcp" in t or "bhp" in t or "esp" in t: return "Pump"
    if "tub" in t or "packer" in t: return "Tubing"
    return "Other"

jobs = pd.read_csv("data/well_jobs.csv", parse_dates=["start_date"])
fails = pd.read_csv("data/production_failures.csv", parse_dates=["failure_start_date"])

# 1. keep only reactive repair jobs
rep = jobs[jobs.job_category.isin(["Rod Truck", "Workover"]) & jobs.job_type.fillna("").str.contains(FAIL_JOB)]

# 2. put both sources into the same shape
from_jobs = pd.DataFrame({"uwi": rep.uwi, "date": rep.start_date,"component": rep.job_type.map(component), "source": "job","cost": rep.field_est_total.clip(lower=0)})   # negative costs -> 0
from_log = pd.DataFrame({"uwi": fails.uwi, "date": fails.failure_start_date,"component": fails.failure_type.map(component), "source": "failure_log","cost": float("nan")})
ev = pd.concat([from_log, from_jobs]).dropna(subset=["date"]).sort_values(["uwi", "date"])

# 3. Merge events < 7 days apart on the same well
gap = ev.groupby("uwi").date.diff().dt.days          # days since this well's previous event

ev["event_id"] = (gap.isna() | (gap > MERGE_DAYS)).cumsum()   # new id = new failure

events = ev.groupby("event_id").agg(uwi=("uwi", "first"), date=("date", "min"),component=("component", lambda s: s.mode().iat[0]),in_failure_log=("source", lambda s: (s == "failure_log").any()),in_jobs=("source", lambda s: (s == "job").any()),repair_cost=("cost", "sum")).reset_index(drop=True)

#4. Keep only producting wells
daily = pd.read_csv("data/daily_production.csv", usecols=["uwi", "flow_direction", "well_daily_status"])
producing_wells = set(daily.loc[(daily.flow_direction == "production") & (daily.well_daily_status == "Producing"), "uwi"])

before = len(events)

events = events[events.uwi.isin(producing_wells)].reset_index(drop=True)
print(f"Dropped {before - len(events)} failures on non-producing wells")

events.to_csv("outputs/failure_events.csv", index=False)

#5. Checks
print(f"{len(events):,} failures on {events.uwi.nunique():,} wells")
print(events.groupby(events.date.dt.year).size())
print(events.component.value_counts())
recent = events[events.date >= "2025-01-01"]
print(f"2025+ failures also found in jobs: {recent.in_jobs.mean():.0%}")