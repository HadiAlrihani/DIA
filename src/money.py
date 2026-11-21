import pandas as pd

# Assumptions (say these out loud on the slide, and change them if you like)
OIL_PRICE = 500    # dollars per m3 of oil
DAYS_SAVED = 5     # days of downtime avoided by reaching a well before it fails
N = 50             # visits per day

t = pd.read_pickle("outputs/test_scored.pkl")

top = t.groupby("date", group_keys=False).apply(lambda d: d.nlargest(N, "priority"))
rand = t.groupby("date", group_keys=False).apply(lambda d: d.sample(min(N, len(d)), random_state=0))

def summary(name, d):
    hits = d[d.y == 1]                                   # visits that reached a well that really failed
    oil_saved = (hits.oil_produced_m3_m30.fillna(0) * DAYS_SAVED).sum()
    print(f"\n{name}")
    print(f"  Visits that hit a failing well: {len(hits)/len(d):.1%}")
    print(f"  Oil protected per 100 visits:   {oil_saved/len(d)*100:,.0f} m3")
    print(f"  Value per 100 visits:           ${oil_saved*OIL_PRICE/len(d)*100:,.0f}")

summary("Our top 50 (model + oil)", top)
summary("Random 50", rand)