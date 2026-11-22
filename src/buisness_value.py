import json
import pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Assumptions
WTI_USD_BBL = 83.21          # WTI settle, Nov 20 2026
WCS_DISCOUNT = 26.0          # WCS heavy oil discount to WTI, Nov 2026
BBL_PER_M3 = 6.29
OIL_PRICE_M3 = (WTI_USD_BBL - WCS_DISCOUNT) * BBL_PER_M3     # ~US$360 per m3
TOP_K = 50                   # wells the team reviews each week

# 1. how much does one failure cost
daily = pd.read_pickle("outputs/daily_clean.pkl")
daily["uwi"] = daily.uwi.astype(str)
events = pd.read_csv("outputs/failure_events.csv", parse_dates=["date"])
events["uwi"] = events.uwi.astype(str)
by_well = {u: w for u, w in daily[["uwi", "production_date", "hours_down", "oil_produced_m3"]].groupby("uwi")}
rows = []
for e in events[events.date >= "2023-01-01"].itertuples():
    w = by_well.get(e.uwi)
    if w is None: continue
    t = w.production_date
    after = w[(t >= e.date - pd.Timedelta(days=3)) & (t <= e.date + pd.Timedelta(days=21))]
    before = w[(t >= e.date - pd.Timedelta(days=60)) & (t < e.date - pd.Timedelta(days=7))]
    if len(before) >= 10:
        rows.append((after.hours_down.sum() / 24, before.oil_produced_m3.mean(), e.repair_cost))
cost = pd.DataFrame(rows, columns=["days_down", "oil_rate", "repair_cost"])
DOWN_DAYS = cost.days_down.mean()
REPAIR = cost.repair_cost.mean()
print(f"1) Cost of a failure (from {len(cost):,} failures since 2023)")
print(f"   downtime: median {cost.days_down.median():.1f} days, mean {DOWN_DAYS:.1f} days")
print(f"   repair cost: median ${cost.repair_cost.median():,.0f}, mean ${REPAIR:,.0f}")
print(f"   oil price assumption: US${WTI_USD_BBL - WCS_DISCOUNT:.0f}/bbl = US${OIL_PRICE_M3:.0f}/m3")
lost_oil = cost.oil_rate * DOWN_DAYS * OIL_PRICE_M3
print(f"   lost oil value per failure: median ${lost_oil.median():,.0f}, "f"top 10% of wells > ${lost_oil.quantile(.9):,.0f}")

# 2. value of the weekly top 50 risk list
d = pd.read_pickle("outputs/test_scores.pkl")
mt = pd.read_pickle("outputs/model_table.pkl")
d = d.merge(mt[["uwi", "date", "oil_produced_m3_m30"]], on=["uwi", "date"], how="left")
d["oil_rate"] = d.oil_produced_m3_m30.fillna(0).clip(lower=0)
d["oil_at_stake"] = d.oil_rate * DOWN_DAYS * OIL_PRICE_M3     # $ of oil lost if it fails
# (repair cost is paid either way, so it is NOT counted as savings)

mon = d[d.date.dt.dayofweek == 0].copy()
mon["fail_date"] = mon.date + pd.to_timedelta(mon.days_to_failure, unit="D")
mon["rank"] = mon.groupby("date").risk.rank(ascending=False, method="first")
weeks = mon.date.nunique(); years = weeks / 52
all_fail = mon[mon.y == 1].groupby(["uwi", "fail_date"]).oil_at_stake.last()      # each failure once
top = mon[mon["rank"] <= TOP_K]
caught = top[top.y == 1].groupby(["uwi", "fail_date"]).oil_at_stake.last()        # flagged at least once
annual = caught.sum() / years
print(f"\n2) Weekly top-{TOP_K} risk list over {weeks} test Mondays")
print(f"   {len(all_fail):,} failures in the test period, ${all_fail.sum() / years / 1e6:.1f}M/yr of oil at stake")
print(f"   list flagged {len(caught):,} of them at least once in the 4-30 days before "f"({len(caught) / len(all_fail):.0%}), covering ${annual / 1e6:.1f}M/yr of oil at stake")

print(f"\n3) Scenario: if early action avoids part of that downtime")
for share in (0.25, 0.5):
    print(f"   avoid {share:.0%}: ${annual * share / 1e6:.1f}M per year saved")

# 3. this weeks watchlist
bundle = pickle.load(open("outputs/model.pkl", "rb"))
feat = pd.read_pickle("outputs/features.pkl")
train_rows = mt[(mt.date >= "2021-04-01") & (mt.date < "2024-07-01")]
for c in ["field_operating_area", "pcp_pump_make", "pcp_pump_model", "last_failed_component"]:
    keep = train_rows[c].astype(str).value_counts()
    keep = keep[keep >= 2000].index
    feat[c] = feat[c].astype(str).where(feat[c].astype(str).isin(keep), "OTHER").astype("category")
today = feat[(feat.date == feat.date.max()) & feat.has_pcp & (feat.hours_on_prod > 0)].copy()
today["risk"] = bundle["model"].predict_proba(today[bundle["features"]])[:, 1]
watch = (today.sort_values("risk", ascending=False).head(TOP_K)[["uwi", "field_operating_area", "risk", "days_since_failure", "well_mtbf","failures_prior_365d", "last_failed_component"]])
watch.insert(0, "priority", range(1, len(watch) + 1))
watch.to_csv("outputs/watchlist_latest.csv", index=False)
print(f"\n4) Watchlist for {feat.date.max():%Y-%m-%d}: top 5 of {len(today)} running wells")
print(watch.head(5).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

# 4. Charts for slides
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,"ytick.color": MUTED})

# (a) lost oil to downtime by year
yr = daily.groupby(daily.production_date.dt.year).lost_oil_m3.sum() / 1000
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(yr.index.astype(str), yr.values, color=BLUE, width=0.6)
bars[-1].set_alpha(0.5)
for b, v in zip(bars, yr.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}k", ha="center", color=INK)
ax.set_ylabel("Oil lost to downtime (thousand m³)")
ax.set_title("Production lost to downtime is rising (2026 = Jan-Oct)", loc="left", color=INK)
ax.grid(axis="y", color="#e5e4df"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figures/lost_oil_by_year.png", dpi=200); plt.close(fig)

# (b) precision of the weekly top-50 list
m = json.load(open("outputs/metrics.json"))
names = ["Random", "History baseline", "Model"]
vals = [m[n]["precision_top50"] * 100 for n in names]
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(names, vals, color=[GRAY, ORANGE, BLUE], height=0.55)
for b, v in zip(bars, vals):
    ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", color=INK)
ax.set_xlabel("Share of weekly top-50 wells that fail within 4-30 days")
ax.set_title(f"Model flags failing wells {vals[2] / vals[0]:.1f}x better than random", loc="left", color=INK)
ax.set_xlim(0, 50); ax.grid(axis="x", color="#e5e4df"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figures/precision_top50.png", dpi=200); plt.close(fig)

# (c) what drives the predictions
imp = pd.read_csv("outputs/feature_importance.csv", index_col=0).pr_auc_drop.clip(lower=0)
def family(f):
    if f in bundle["history"] or f == "last_repair_cost": return "Failure history"
    if f in ("field_operating_area", "pcp_pump_make", "pcp_pump_model", "pump_displacement",
             "rated_throughput", "well_age_yrs", "month"): return "Well / pump facts"
    if "down" in f or "torqued" in f: return "Recent downtime"
    if f.startswith(("oil_", "total_fluid", "water_cut")): return "Production volumes"
    return "Pump sensors"
fam = (imp.groupby(imp.index.map(family)).sum() / imp.sum() * 100).sort_values()
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(fam.index, fam.values, color=BLUE, height=0.55)
for b, v in zip(bars, fam.values):
    ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", color=INK)
ax.set_xlabel("Share of model importance (%)")
ax.set_title("Failure history drives predictions; sensors add little", loc="left", color=INK)
ax.grid(axis="x", color="#e5e4df"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figures/importance_by_family.png", dpi=200); plt.close(fig)

print("\nSaved watchlist_latest.csv and 3 charts in figures/")