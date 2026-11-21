import pandas as pd

HORIZON = 30    # predict failures up to 30 days ahead
BLACKOUT = 3    # but at least 3 days ahead (otherwise it's too late to act)

X = pd.read_pickle("outputs/features.pkl")
events = pd.read_csv("outputs/failure_events.csv", parse_dates=["date"])
events["uwi"] = events.uwi.astype(str)
print(f"Loaded {len(X):,} well-days")

# 1. For each well, find the next failure strictly after today
key = X[["uwi", "date"]].reset_index().sort_values("date")
nxt = pd.merge_asof(key, events[["uwi", "date"]].rename(columns={"date": "next_fail"}).sort_values("next_fail"),left_on="date", right_on="next_fail", by="uwi",direction="forward", allow_exact_matches=False).set_index("index").sort_index()
X["days_to_failure"] = (nxt.next_fail - X.date).dt.days        # NaN = no later failure

#2. the target
X["y"] = ((X.days_to_failure > BLACKOUT) & (X.days_to_failure <= HORIZON)).astype("int8")

# 3. remove the rows we cant use
n0 = len(X)
X = X[X.has_pcp]                                   ; n1 = len(X)   # PCP wells only
X = X[X.hours_on_prod > 0]                         ; n2 = len(X)   # well actually running
X = X[~(X.days_to_failure <= BLACKOUT)]            ; n3 = len(X)   # failure already under way
last_day = X.date.max()
X = X[X.date <= last_day - pd.Timedelta(days=HORIZON)]; n4 = len(X)  # future not known yet
print(f"  removed {n0-n1:>9,} rows on wells with no PCP data")
print(f"  removed {n1-n2:>9,} rows where the well was down all day")
print(f"  removed {n2-n3:>9,} rows 1-{BLACKOUT} days before a failure")
print(f"  removed {n3-n4:>9,} rows in the last {HORIZON} days (outcome unknown)")

X = X.reset_index(drop=True)
X.to_pickle("outputs/model_table.pkl")

# 4. checks
print(f"\nModel table: {len(X):,} rows, {X.uwi.nunique():,} wells")
print(f"Rows labelled y=1 (fails in 4-30 days): {X.y.sum():,} = {X.y.mean():.1%}")
print("\nPositive rate by year:")
print(X.groupby(X.date.dt.year).y.mean().map("{:.1%}".format).to_string())
print("\nOne well in the 35 days before a failure (every 4th day):")
row = X[X.days_to_failure == 4].iloc[1000]          # some row 4 days before a failure
fail_day = row.date + pd.Timedelta(days=4)
show = X[(X.uwi == row.uwi) & (X.date < fail_day) & (X.date >= fail_day - pd.Timedelta(days=35))]
print(show[["date", "days_to_failure", "y"]].iloc[::4].to_string(index=False))