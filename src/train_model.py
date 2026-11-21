import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

X = pd.read_pickle("outputs/model_table.pkl")

# Time split: train on older data, test on the last 90 days (30-day gap so labels don't overlap)
last = X.date.max()
test_start = last - pd.Timedelta(days=90)
train = X[X.date < test_start - pd.Timedelta(days=30)]
test = X[X.date >= test_start].copy()

DROP = ["uwi", "date", "y", "days_to_failure", "hours_on_prod", "oil_m3", "has_pcp"]
feats = [c for c in X.columns if c not in DROP]

model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                           num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                           class_weight="balanced", verbose=-1)
model.fit(train[feats], train.y)

test["p"] = model.predict_proba(test[feats])[:, 1]
print("AUC:", round(roc_auc_score(test.y, test.p), 3))
print("Avg precision:", round(average_precision_score(test.y, test.p), 3),
      "(random would be", round(test.y.mean(), 3), ")")

# Priority = chance of failure x recent oil per day
test["priority"] = test.p * test.oil_produced_m3_m30.fillna(0)

# Top-50 wells per day vs random
N = 50
top = test.groupby("date", group_keys=False).apply(lambda d: d.nlargest(N, "priority"))
print(f"Top {N} by priority: {top.y.mean():.1%} actually fail in 4-30 days")
print(f"All wells (random): {test.y.mean():.1%}")

# Which signs matter (for the 'indicators' slide)
imp = pd.Series(model.feature_importances_, feats).sort_values(ascending=False)
print(imp.head(15))
test.to_pickle("outputs/test_scored.pkl")