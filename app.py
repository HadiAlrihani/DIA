import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from scipy.spatial import ConvexHull

st.set_page_config(page_title="PCP Failure Watchlist", layout="wide")

# 1. load the data
@st.cache_data
def load():
    weekly = pd.read_csv("outputs/dash_weekly.csv", parse_dates=["date"], dtype={"uwi": str})
    wells = pd.read_csv("outputs/dash_wells.csv", dtype={"uwi": str})
    fails = pd.read_csv("outputs/dash_failures.csv", parse_dates=["date"], dtype={"uwi": str})
    return weekly, wells, fails


weekly, wells, fails = load()
dates = sorted(weekly.date.unique(), reverse=True)
latest = dates[0]

# 2. filters
st.sidebar.title("Filters")
date = st.sidebar.selectbox(
    "Week", dates,
    format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d") + ("  (latest)" if d == latest else ""))
fields_this_week = weekly[weekly.date == date].field_operating_area.dropna().unique()
field = st.sidebar.selectbox("Field", ["All fields"] + sorted(fields_this_week))
top_k = st.sidebar.slider("Wells the team can review this week (top K)", 10, 200, 50, step=10)
st.sidebar.subheader("Map")
basemap = st.sidebar.radio("Basemap", ["Dark", "Dark (no labels)", "Streets (light)"])
show_fields = st.sidebar.checkbox("Show field areas and names", value=True)
show_heat = st.sidebar.checkbox("Show risk heatmap", value=False)
st.sidebar.caption("Risk = model probability that the well has a rod / pump / tubing ""failure 4-30 days after this date. Ranking is across all fields.")

week = weekly[weekly.date == date].merge(wells[["uwi", "lat", "lon"]], on="uwi", how="inner")
week["flagged"] = week["rank"] <= top_k
view = week if field == "All fields" else week[week.field_operating_area == field]
outcome_known = week.y.notna().any()

# 3. headline numbers
st.title("PCP Failure Watchlist")
st.caption(f"Week of {pd.Timestamp(date):%B %d, %Y} · {field}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Running wells", f"{len(view):,}")
c2.metric(f"Flagged (top {top_k})", f"{int(view.flagged.sum()):,}")
c3.metric("Average risk", f"{view.risk.mean():.0%}")
past = weekly[weekly.y.notna()]
hist_rate = past[past["rank"] <= top_k].y.mean()
if outcome_known and view.flagged.any():
    c4.metric("Flagged wells that failed", f"{view[view.flagged].y.mean():.0%}",help=f"Average over all test weeks at top {top_k}: {hist_rate:.0%}")
else:
    c4.metric("Expected hit rate", f"{hist_rate:.0%}",help="Outcome not known yet; this is the average over the 2025-26 test weeks")

# 4. map and side panel
left, right = st.columns([3, 2])

def risk_colour(r):
    """Dim slate (low risk) -> bright orange-red (high risk), readable on a dark map."""
    t = min(max(r / 0.5, 0.0), 1.0)
    lo, hi = np.array([70, 95, 120, 150]), np.array([255, 90, 60, 240])
    return (lo + (hi - lo) * t).astype(int).tolist()


BASEMAPS = {
    "Dark": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    "Dark (no labels)": "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json",
    "Streets (light)": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
}


def field_areas(df):
    """Outline each field with the convex hull of its wells, labelled with flagged count."""
    rows = []
    for name, g in df.groupby("field_operating_area"):
        pts = g[["lon", "lat"]].drop_duplicates().to_numpy()
        label = f"{name} ({int(g.flagged.sum())} flagged)"
        centre = [float(g.lon.mean()), float(g.lat.mean())]
        if len(pts) >= 3:
            try:
                hull = pts[ConvexHull(pts).vertices]
                pad = (hull - hull.mean(axis=0)) * 0.08 + hull     # grow 8% so edge wells sit inside
                rows.append({"name": label, "polygon": pad.tolist(), "centre": centre})
                continue
            except Exception:
                pass                                                # wells in a straight line
        rows.append({"name": label, "polygon": None, "centre": centre})
    return pd.DataFrame(rows)


view = view.assign(
    colour=view.risk.map(risk_colour),
    risk_txt=(view.risk * 100).round(0).astype(int).astype(str) + "%",
    mtbf_txt=view.well_mtbf.round(0).astype("Int64").astype(str).replace("<NA>", "n/a"),
    dsf_txt=view.days_since_failure.round(0).astype("Int64").astype(str).replace("<NA>", "n/a"),
    f365_txt=view.failures_prior_365d.fillna(0).astype(int).astype(str))

with left:
    st.subheader("Where the risk is")
    layers = []
    if show_fields and len(view):
        areas = field_areas(view)
        poly = areas[areas.polygon.notna()]
        layers.append(pdk.Layer("PolygonLayer", data=poly, get_polygon="polygon",
                                get_fill_color=[57, 135, 229, 25], get_line_color=[57, 135, 229, 160],
                                line_width_min_pixels=1, stroked=True, filled=True, pickable=False))
    if show_heat:
        layers.append(pdk.Layer("HeatmapLayer", data=view, get_position=["lon", "lat"],
                                get_weight="risk", radius_pixels=45, opacity=0.45, threshold=0.05))
    layers += [
        pdk.Layer("ScatterplotLayer", data=view[~view.flagged], get_position=["lon", "lat"],
                  get_fill_color="colour", get_radius=200, radius_min_pixels=2, pickable=True),
        pdk.Layer("ScatterplotLayer", data=view[view.flagged], get_position=["lon", "lat"],
                  get_fill_color="colour", get_radius=500, radius_min_pixels=6, stroked=True,
                  get_line_color=[255, 255, 255, 255], line_width_min_pixels=2, pickable=True),
    ]
    if show_fields and len(view):
        layers.append(pdk.Layer("TextLayer", data=areas, get_position="centre", get_text="name",
                                get_size=13, get_color=[240, 239, 236, 230],
                                get_alignment_baseline="'bottom'", pickable=False))
    centre = view[view.flagged] if view.flagged.any() else view
    if centre.empty:
        centre = wells          # no wells in this selection: centre on all wells
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=float(centre.lat.median()),
                                         longitude=float(centre.lon.median()),
                                         zoom=7 if field == "All fields" else 10),
        map_provider="carto",
        map_style=BASEMAPS[basemap],
        tooltip={"html": "<b>{field_operating_area}</b> · {uwi}<br/>Risk: {risk_txt} (rank {rank})"
                         "<br/>Days since last failure: {dsf_txt}<br/>Well MTBF: {mtbf_txt} days"
                         "<br/>Failures in past year: {f365_txt}",
                 "style": {"backgroundColor": "#1f1f1e", "color": "#f0efec", "fontSize": "13px"}})
    st.pydeck_chart(deck, height=620)
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;font-size:13px;color:#c3c2b7'>"
        "<span>Risk</span><span>0%</span>"
        "<span style='width:160px;height:10px;border-radius:5px;background:linear-gradient(90deg,"
        "rgb(70,95,120),rgb(255,90,60))'></span><span>50%+</span>"
        "<span style='margin-left:14px'>⚪ white outline = this week's watchlist</span></div>",
        unsafe_allow_html=True)

with right:
    st.subheader("Flagged wells by field")
    by_field = (week[week.flagged].groupby("field_operating_area").size()
                .sort_values(ascending=False).rename("flagged wells"))
    bars = alt.Chart(by_field.reset_index()).mark_bar(color="#3987e5").encode(
        x=alt.X("flagged wells:Q", title="Flagged wells"),
        y=alt.Y("field_operating_area:N", sort="-x", title=None),
        tooltip=["field_operating_area:N", "flagged wells:Q"])
    st.altair_chart(bars, use_container_width=True)

    st.subheader("Watchlist")
    table = view[view.flagged].sort_values("rank")
    if outcome_known:
        outcome = table.y.map({1.0: "Failed (4-30 d)", 0.0: "No failure"})
    else:
        outcome = pd.Series("Pending", index=table.index)
    st.dataframe(pd.DataFrame({
        "Rank": table["rank"], "Field": table.field_operating_area, "Well": table.uwi,
        "Risk": table.risk_txt, "Days since failure": table.dsf_txt, "MTBF (days)": table.mtbf_txt,
        "Failures past year": table.f365_txt, "Last failed": table.last_failed_component,
        "Outcome": outcome}), hide_index=True, height=380)
    
# 5. drill one well over time
st.subheader("Well history")
options = table.uwi.tolist() or view.sort_values("rank").uwi.head(20).tolist()
if options:
    uwi = st.selectbox("Choose a well (watchlist wells listed first)", options)
    hist = weekly[weekly.uwi == uwi][["date", "risk"]]
    f = fails[fails.uwi == uwi]
    line = alt.Chart(hist).mark_line(point=True).encode(
        x=alt.X("date:T", title="Week"), y=alt.Y("risk:Q", title="Failure risk", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("risk:Q", format=".0%")])
    rules = alt.Chart(f).mark_rule(color="#e66767", strokeWidth=2).encode(
        x="date:T", tooltip=[alt.Tooltip("date:T", title="Failure"), "component:N"])
    st.altair_chart(line + rules, use_container_width=True)
    st.caption(f"Red lines = actual failures ({len(f)} since Jan 2025).")

# 6. how well did the model perform
with st.expander("Model track record (2025-26 test weeks)"):
    perf = (past.assign(hit=past.y.where(past["rank"] <= top_k))      # outcome of flagged wells only
            .groupby("date").agg(**{"Flagged wells that failed": ("hit", "mean"),
                                    "Random pick (base rate)": ("y", "mean")})
            .reset_index().melt("date", var_name="series", value_name="rate"))
    chart = alt.Chart(perf).mark_line().encode(
        x=alt.X("date:T", title="Week"), y=alt.Y("rate:Q", title="Share that failed in 4-30 days",
                                                  axis=alt.Axis(format="%")),
        color=alt.Color("series:N", scale=alt.Scale(range=["#3987e5", "#8a8984"]),
                        legend=alt.Legend(orient="bottom", title=None)))
    st.altair_chart(chart, use_container_width=True)
    st.write(f"At top {top_k}: **{hist_rate:.0%}** of flagged wells failed within 4-30 days, "
             f"vs **{past.y.mean():.0%}** for a random pick.")