import pandas as pd
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

X = pd.read_pickle("outputs/model_table.pkl")

# 1. Time based split
TRAIN_START, VAL_START, TEST_START = "2021-04-01", "2024-07-01", "2025-01-01"
X = X[X.date >= TRAIN_START].reset_index(drop=True)      # skip first 3 months (history warming up)
train = X.date < VAL_START
val = (X.date >= VAL_START) & (X.date < TEST_START)
test = X.date >= TEST_START
for name, m in [("train", train), ("val", val), ("test", test)]:
    d = X[m]
    print(f"{name:5s} {d.date.min():%Y-%m-%d} to {d.date.max():%Y-%m-%d}  "f"{m.sum():>9,} rows  positive rate {d.y.mean():.1%}")

# 2. which columns the model will see
NOT_FEATURES = ["uwi", "date", "hours_on_prod", "oil_m3", "has_pcp","days_to_failure", "y"]             # IDs, and the ANSWER (never a feature!)
FEATURES = [c for c in X.columns if c not in NOT_FEATURES]
HISTORY = ["days_since_failure", "failures_prior_365d", "well_mtbf", "last_run_life","run_life_vs_mtbf", "n_failures_to_date", "last_failed_component"]

# rare categories -> "OTHER" (the model allows at most 255 categories per column)
for c in ["field_operating_area", "pcp_pump_make", "pcp_pump_model", "last_failed_component"]:
    counts = X.loc[train, c].astype(str).value_counts()
    keep = counts[counts >= 2000].index
    X[c] = X[c].astype(str).where(X[c].astype(str).isin(keep), "OTHER").astype("category")
print(f"\n{len(FEATURES)} features; history-only baseline uses {len(HISTORY)}")

# 3. Train, choosing the number of trees on the validation period
def train_model(cols, label):
    settings = dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,l2_regularization=1.0, categorical_features="from_dtype",early_stopping=False, random_state=42)
    model = HistGradientBoostingClassifier(max_iter=25, warm_start=True, **settings)
    best_trees, best_score = None, -1
    print(f"\n{label}")
    for trees in [25, 50, 100, 200]:
        model.set_params(max_iter=trees)
        model.fit(X.loc[train, cols], X.y[train])          # warm_start: adds trees, doesn't restart
        p = model.predict_proba(X.loc[val, cols])[:, 1]
        score = average_precision_score(X.y[val], p)
        print(f"  {trees:3d} trees -> validation PR-AUC {score:.3f}  ROC-AUC {roc_auc_score(X.y[val], p):.3f}")
        if score > best_score:
            best_trees, best_score = trees, score
    print(f"  best: {best_trees} trees. Refitting on train + validation...")
    final = HistGradientBoostingClassifier(max_iter=best_trees, **settings)
    final.fit(X.loc[train | val, cols], X.y[train | val])
    return final

baseline = train_model(HISTORY, "History-only baseline")
model = train_model(FEATURES, "Full model")

# 4. score and save
out = X.loc[test, ["uwi", "date", "y", "days_to_failure", "oil_m3"]].copy()
out["risk"] = model.predict_proba(X.loc[test, FEATURES])[:, 1]
out["risk_baseline"] = baseline.predict_proba(X.loc[test, HISTORY])[:, 1]
out.to_pickle("outputs/test_scores.pkl")
pickle.dump({"model": model, "baseline": baseline, "features": FEATURES, "history": HISTORY},
            open("outputs/model.pkl", "wb"))
print(f"\nSaved model.pkl and test_scores.pkl ({len(out):,} test rows)")