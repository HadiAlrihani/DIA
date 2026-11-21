import numpy as np
import pandas as pd

# 1. load only the columns from the producing days
TEXT = ["uwi", "field_operating_area", "province", "well_daily_status", "flow_direction","downtime_code_2", "downtime_code_3", "pcp_pump_make", "pcp_pump_model","production_date", "on_production_date"]
NUM = ["hours_down", "hours_on_prod", "lost_oil_m3", "oil_produced_m3", "water_produced_m3","gas_produced_e3m3", "tubing_pressure_kpa", "casing_pressure_kpa", "pump_efficiency_pct","pcp_pump_displacement_m3d100rpm", "pcp_pump_rpm", "pcp_pump_electric_current_a","pcp_pump_rod_torque_ftlb", "pcp_pump_volume_throughput_m3d"]

dtypes = {c: "str" for c in TEXT} | {c: "float32" for c in NUM}
parts = []

for chunk in pd.read_csv("data/daily_production.csv", usecols=TEXT + NUM,
                         dtype=dtypes, chunksize=500_000):
    chunk = chunk[(chunk.flow_direction == "production") &
                  (chunk.well_daily_status == "Producing")]
    parts.append(chunk)

df = pd.concat (parts, ignore_index=True)
df["production_date"] = pd.to_datetime(df.production_date)
df["on_production_date"] = pd.to_datetime(df.on_production_date, errors="coerce")
print(f"Producing well-days loaded: {len(df):,} wells: {df.uwi.nunique():,}")

# 2. 1 row per well per day, in time order
before = len(df)
df = df.drop_duplicates(["uwi", "production_date"])
df = df.sort_values(["uwi", "production_date"]).reset_index(drop=True)
print(f"Duplicate well-days removed: {before - len(df)}")

# 3. impossible values
LIMITS = {                                   # max physically believable value
    "pcp_pump_rpm": 700,                     # PCPs run between 50-500 RPM
    "pcp_pump_electric_current_a": 400,      # amps
    "pcp_pump_rod_torque_ftlb": 8000,        # ft-lb
    "pump_efficiency_pct": 3,                # stored as decimal; 91 = data error
    "pcp_pump_volume_throughput_m3d": 1500,  # m3/day
    "pcp_pump_displacement_m3d100rpm": 1100,
    "tubing_pressure_kpa": 20000,            # catches 999999 placeholders
    "casing_pressure_kpa": 20000,
}

print("\nValues set to NaN:")

for col, hi in LIMITS.items():
    bad = (df[col] < 0) | (df[col] > hi)
    print(f"  {col:34s} {bad.sum():>7,}")
    df.loc[bad, col] = np.nan

# 4. fix the inconsistent labels
df["pcp_pump_make"] = df.pcp_pump_make.str.upper().str.strip()   # 'Mantl' -> 'MANTL'

# 5. new columns built from the existing one
df["total_fluid_m3"] = df.oil_produced_m3 + df.water_produced_m3
df["water_cut"] = df.water_produced_m3 / df.total_fluid_m3.replace(0, np.nan)
df["torque_per_rpm"] = df.pcp_pump_rod_torque_ftlb / df.pcp_pump_rpm.replace(0, np.nan)

code = df.downtime_code_3.fillna("").str.lower()
lift = code.str.contains("rod|torqued|pump|tubing|downhole")
df["lift_downtime_hrs"] = np.where(lift, df.hours_down, 0.0)      # hours down for lift problems
df["torqued_flag"] = code.str.contains("torqued").astype("float32")  # 1 if 'Well Torqued' that day

# 6. save it
for c in ["uwi", "field_operating_area", "province", "pcp_pump_make", "pcp_pump_model","downtime_code_2", "downtime_code_3", "well_daily_status", "flow_direction"]:
    df[c] = df[c].astype("category")                              # saves lots of memory

df.to_pickle("outputs/daily_clean.pkl")

print(f"\nSaved {len(df):,} rows, {df.shape[1]} columns, "f"{df.memory_usage(deep=True).sum() / 1e6:.0f} MB in memory")
print(f"Wells with any PCP RPM data: {df.loc[df.pcp_pump_rpm.notna(), 'uwi'].nunique():,}")
print(f"Median water cut: {df.water_cut.median():.3f}")
print(df.pcp_pump_make.value_counts().head(6))