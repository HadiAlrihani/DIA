import pandas as pd 
import json
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

d = pd.read_pickle("outputs/test_scores.pkl")
d["risk_random"] = np.random.default_rng(0).random(len(d))
SCORES = {"Model": "risk", "History baseline": "risk_baseline", "Random": "risk_random"}
print(f"Test period: {d.date.min():%Y-%m-%d} to {d.date.max():%Y-%m-%d}, "f"{len(d):,} well-days, positive rate {d.y.mean():.1%}")

# 1. overall ranking
print("\n1) Overall ranking")
print(f"{'':18s}{'ROC-AUC':>9s}{'PR-AUC':>9s}")
for name, col in SCORES.items():
    print(f"{name:18s}{roc_auc_score(d.y, d[col]):9.3f}{average_precision_score(d.y, d[col]):9.3f}")

# 2. weekly ranking and how an operator would use it
mondays = d[d.date.dt.dayofweek == 0].copy()
for col in SCORES.values():
    mondays[f"rank_{col}"] = mondays.groupby("date")[col].rank(ascending=False, method="first")
print(f"\n2) Weekly review: every Monday flag the top K wells ({mondays.date.nunique()} Mondays, "
      f"~{mondays.groupby('date').size().mean():.0f} running wells each)")
print("   Precision = share of flagged wells that fail within 4-30 days")
print(f"{'':18s}" + "".join(f"{'top ' + str(k):>10s}" for k in (25, 50, 100)))
results = {}
for name, col in SCORES.items():
    precs = [mondays[mondays[f"rank_{col}"] <= k].y.mean() for k in (25, 50, 100)]
    results[name] = {f"precision_top{k}": p for k, p in zip((25, 50, 100), precs)}
    print(f"{name:18s}" + "".join(f"{p:10.1%}" for p in precs))
lift = results["Model"]["precision_top50"] / d.y.mean()
print(f"   -> Top-50 list is {lift:.1f}x better than picking wells at random")

# 3. failures caught and a single decision per failure
# for each failure, look at the LAST Monday review 4-30 days before it. Was the well on the list?
pos = mondays[mondays.y == 1].copy()
pos["fail_date"] = pos.date + pd.to_timedelta(pos.days_to_failure, unit="D")
last_review = pos.sort_values("date").groupby(["uwi", "fail_date"]).tail(1)
print(f"\n3) Failures caught in advance ({len(last_review):,} failures; last weekly review before each)")
print(f"{'':18s}" + "".join(f"{'top ' + str(k):>10s}" for k in (25, 50, 100)))
for name, col in SCORES.items():
    rec = [(last_review[f"rank_{col}"] <= k).mean() for k in (25, 50, 100)]
    results[name].update({f"recall_top{k}": r for k, r in zip((25, 50, 100), rec)})
    print(f"{name:18s}" + "".join(f"{r:10.1%}" for r in rec))
caught = last_review[last_review.rank_risk <= 50]
print(f"   -> Top-50 list catches {len(caught):,} failures, median warning "
      f"{caught.days_to_failure.median():.0f} days")

# 4. which features matters the most?
bundle = pickle.load(open("outputs/model.pkl", "rb"))
X = pd.read_pickle("outputs/model_table.pkl")
X = X[X.date >= "2021-04-01"].reset_index(drop=True)
train = X.date < "2024-07-01"
for c in ["field_operating_area", "pcp_pump_make", "pcp_pump_model", "last_failed_component"]:
    counts = X.loc[train, c].astype(str).value_counts()          # SAME prep as train.py
    keep = counts[counts >= 2000].index
    X[c] = X[c].astype(str).where(X[c].astype(str).isin(keep), "OTHER").astype("category")
sample = X[X.date >= "2025-01-01"].sample(50_000, random_state=1)
pi = permutation_importance(bundle["model"], sample[bundle["features"]], sample.y,scoring="average_precision", n_repeats=3, random_state=0)
imp = pd.Series(pi.importances_mean, index=bundle["features"]).sort_values(ascending=False)

def family(f):
    if f in bundle["history"] or f in ("last_repair_cost",): return "Failure history"
    if f in ("field_operating_area", "pcp_pump_make", "pcp_pump_model", "pump_displacement","rated_throughput", "well_age_yrs", "month"): return "Well / pump facts"
    if "down" in f or "torqued" in f: return "Recent downtime"
    if f.startswith(("oil_", "total_fluid", "water_cut")): return "Production volumes"
    return "Pump sensors"
print("\n4) Top 12 features (drop in PR-AUC when shuffled)")
print(imp.head(12).round(4).to_string())
fam = imp.clip(lower=0).groupby(imp.index.map(family)).sum()
print("\n   Share of importance by feature family:")
print((fam / fam.sum()).sort_values(ascending=False).map("{:.0%}".format).to_string())

imp.to_csv("outputs/feature_importance.csv", header=["pr_auc_drop"])
json.dump(results, open("outputs/metrics.json", "w"), indent=2)
print("\nSaved metrics.json and feature_importance.csv")
