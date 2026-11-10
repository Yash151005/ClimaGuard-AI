"""
ClimaGuard AI — Federated Multimodal Hyperlocal Disaster Early Warning
Single-file Streamlit app using LIVE public data (no API keys needed):
  • Open-Meteo Forecast API  → 72h hyperlocal forecast for every ~1 km grid cell
  • Open-Meteo Archive API   → 2016-present daily history (ERA5 reanalysis) for climatology + ML
  • NASA EONET v3 API        → open floods / landslides / severe storms / temperature extremes

Run:
    pip install streamlit plotly pandas numpy requests
    streamlit run climaguard_app.py
"""

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG & CONSTANTS
# ----------------------------------------------------------------------------
st.set_page_config(page_title="ClimaGuard AI", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")

TZ = "Asia/Kolkata"
IST = ZoneInfo(TZ)
HEAVY_RAIN_MM = 64.5          # IMD "heavy rain" threshold (mm/day)
EVENT_RADIUS_KM = 250
FEATS = ["rain_today", "rain_3day", "tmax_today", "sin_month", "cos_month", "bias"]

CITIES = {
    "Pune, Maharashtra": (18.5204, 73.8567),
    "Mumbai, Maharashtra": (19.0760, 72.8777),
    "Ahmedabad, Gujarat": (23.0225, 72.5714),
    "Bengaluru, Karnataka": (12.9716, 77.5946),
    "Delhi": (28.6139, 77.2090),
    "Kolkata, West Bengal": (22.5726, 88.3639),
    "Chennai, Tamil Nadu": (13.0827, 80.2707),
    "Guwahati, Assam": (26.1445, 91.7362),
    "Jaipur, Rajasthan": (26.9124, 75.7873),
}

# Three simulated "State DM nodes" for federated learning (raw data never pooled)
NODES = [
    ("Maharashtra node", "Pune", 18.5204, 73.8567),
    ("Gujarat node", "Ahmedabad", 23.0225, 72.5714),
    ("Karnataka node", "Bengaluru", 12.9716, 77.5946),
]

LEVEL_COLORS = {
    "🟢 Safe": "#16a34a",
    "🟡 Watch": "#eab308",
    "🟠 Warning": "#f97316",
    "🔴 Emergency": "#dc2626",
}

CSS = """
<style>
.stApp { background:#000000; color:#f8fafc; }
section[data-testid="stMain"] { background:#000000; }
section[data-testid="stSidebar"] { background:#050505; border-right:1px solid #262626; }
h1,h2,h3,h4,h5,h6,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"] { color:#f8fafc !important; }
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label { color:#e5e7eb !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#d1d5db !important; }
.stTextInput input, .stNumberInput input,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div { background:#111111; color:#f8fafc; border-color:#404040; }
button, [data-baseweb="tab"] { color:#f8fafc !important; }
.stTabs [data-baseweb="tab-list"] { background:#000000; }
.stTabs [data-baseweb="tab-highlight"] { background:#38bdf8; }
.stDataFrame, [data-testid="stDataFrame"] { background:#111111; }
.hero { background: linear-gradient(90deg,#2563eb,#7c3aed,#db2777); padding:22px 28px;
        border-radius:20px; color:white; box-shadow:0 10px 30px rgba(37,99,235,.25); margin-bottom:18px; }
.hero h1 { color:white !important; margin:0; font-size:2.1rem; }
.hero p { margin:4px 0 0 0; opacity:.95; }
.kpi { background:#111111; border-radius:18px; padding:16px 18px; box-shadow:0 6px 18px rgba(0,0,0,.35);
       border-top:6px solid var(--c); min-height:120px; }
.kpi .label { color:#cbd5e1; font-size:.8rem; font-weight:700; text-transform:uppercase; letter-spacing:.5px; }
.kpi .value { font-size:1.8rem; font-weight:800; color:var(--c); }
.kpi .sub { color:#cbd5e1; font-size:.85rem; }
.stTabs [data-baseweb="tab"] { font-weight:700; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

HTTP = requests.Session()


# ----------------------------------------------------------------------------
# DATA ACCESS (all real, live public APIs)
# ----------------------------------------------------------------------------
def get_json(url, params):
    r = HTTP.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()


def make_grid(lat, lon, k):
    """(2k+1)x(2k+1) grid of ~1 km cells centred on the city."""
    dlat = 0.00901                                  # ≈ 1 km in latitude
    dlon = dlat / math.cos(math.radians(lat))
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            rows.append({"cell": f"R{i + k + 1}-C{j + k + 1}",
                         "lat": lat + i * dlat, "lon": lon + j * dlon})
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecast(lats: tuple, lons: tuple):
    """Hourly forecast incl. past 7 days (antecedent rainfall) + next 4 days."""
    params = {
        "latitude": ",".join(f"{x:.4f}" for x in lats),
        "longitude": ",".join(f"{x:.4f}" for x in lons),
        "hourly": "temperature_2m,apparent_temperature,precipitation,relative_humidity_2m",
        "past_days": 7, "forecast_days": 4, "timezone": TZ,
    }
    data = get_json("https://api.open-meteo.com/v1/forecast", params)
    if isinstance(data, dict):
        data = [data]
    out = []
    for loc in data:
        df = pd.DataFrame(loc["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        out.append(df)
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_archive(lat, lon, start="2016-01-01"):
    """Daily historical rainfall & max temperature (ERA5 reanalysis)."""
    end = (datetime.now(IST) - timedelta(days=7)).strftime("%Y-%m-%d")
    data = get_json("https://archive-api.open-meteo.com/v1/archive", {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "daily": "precipitation_sum,temperature_2m_max", "timezone": TZ,
    })
    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df.dropna().reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_events():
    """NASA EONET open natural events (satellite-observed floods, landslides, storms, heat)."""
    data = get_json("https://eonet.gsfc.nasa.gov/api/v3/events", {
        "status": "open", "limit": 300,
        "category": "floods,landslides,severeStorms,tempExtremes",
    })
    rows = []
    for ev in data.get("events", []):
        geo = ev.get("geometry", [])
        if not geo or geo[-1].get("type") != "Point":
            continue
        lon, lat = geo[-1]["coordinates"][:2]
        rows.append({
            "title": ev["title"],
            "category": ev["categories"][0]["title"],
            "lat": lat, "lon": lon,
            "date": geo[-1]["date"][:10],
            "link": ev.get("link", ""),
        })
    return pd.DataFrame(rows, columns=["title", "category", "lat", "lon", "date", "link"])


# ----------------------------------------------------------------------------
# CLIMATOLOGY & HELPERS
# ----------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R, p = 6371.0, math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def build_climatology(arc, month):
    d = arc.sort_values("time").reset_index(drop=True).copy()
    d["p3"] = d["precipitation_sum"].rolling(3).sum()
    d["p10"] = d["precipitation_sum"].rolling(10).sum()
    m = d[d["time"].dt.month == month].dropna()
    return {
        "p3_p95": float(m["p3"].quantile(0.95)),
        "p10_p95": float(m["p10"].quantile(0.95)),
        "tmax_p90": float(m["temperature_2m_max"].quantile(0.90)),
    }


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def auc(y, s):
    """ROC-AUC via Mann-Whitney rank statistic."""
    y = np.asarray(y); s = np.asarray(s)
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def make_xy(arc):
    d = arc.sort_values("time").reset_index(drop=True).copy()
    d["p3"] = d["precipitation_sum"].rolling(3).sum()
    nxt = d["precipitation_sum"].shift(-1)                  # predict TOMORROW's heavy rain
    d["y"] = (nxt >= HEAVY_RAIN_MM).astype(float).where(nxt.notna())
    d = d.dropna(subset=["p3", "y"])
    m = d["time"].dt.month.values
    X = np.column_stack([
        d["precipitation_sum"].values / 50,
        d["p3"].values / 100,
        (d["temperature_2m_max"].values - 30) / 10,
        np.sin(2 * np.pi * m / 12),
        np.cos(2 * np.pi * m / 12),
        np.ones(len(d)),
    ])
    return X, d["y"].values, d["time"].dt.year.values


def feat_now(p24, p72, month):
    return np.array([
        p24.precipitation.sum() / 50,
        p72.precipitation.sum() / 100,
        (p24.temperature_2m.max() - 30) / 10,
        math.sin(2 * math.pi * month / 12),
        math.cos(2 * math.pi * month / 12),
        1.0,
    ])


# ----------------------------------------------------------------------------
# FEDERATED LEARNING (FedAvg): raw data stays in each node, only weights move
# ----------------------------------------------------------------------------
def local_train(X, y, w0, steps=200, lr=0.8):
    w = w0.copy()
    for _ in range(steps):
        p = sig(X @ w)
        w -= lr * (X.T @ (p - y)) / len(y)
    return w


@st.cache_data(ttl=86400, show_spinner=False)
def run_federated(rounds=6):
    last_year = datetime.now(IST).year - 1
    nodes = []
    for name, city, la, lo in NODES:
        arc = fetch_archive(la, lo)
        X, y, yr = make_xy(arc)
        tr, te = yr < last_year, yr == last_year
        nodes.append(dict(name=name, city=city, Xtr=X[tr], ytr=y[tr], Xte=X[te], yte=y[te]))

    w = np.zeros(len(FEATS))
    history = []
    for r in range(1, rounds + 1):
        local_ws, sizes = [], []
        for n in nodes:                                    # each node trains locally
            local_ws.append(local_train(n["Xtr"], n["ytr"], w))
            sizes.append(len(n["ytr"]))
        w = np.average(np.array(local_ws), axis=0, weights=np.array(sizes, dtype=float))  # server
        Xte = np.vstack([n["Xte"] for n in nodes]); yte = np.concatenate([n["yte"] for n in nodes])
        history.append({"round": r, "test_auc": auc(yte, sig(Xte @ w))})

    node_rows = []
    for n in nodes:
        node_rows.append({
            "State node": n["name"], "City": n["city"],
            "Train days": len(n["ytr"]),
            "Heavy-rain days (train)": int(n["ytr"].sum()),
            "Local test AUC": round(auc(n["yte"], sig(n["Xte"] @ w)), 3),
        })

    Xte = np.vstack([n["Xte"] for n in nodes]); yte = np.concatenate([n["yte"] for n in nodes])
    thr = 0.15
    pred = sig(Xte @ w) >= thr
    actual = yte == 1
    tp = float((pred & actual).sum()); fp = float((pred & ~actual).sum()); fn = float((~pred & actual).sum())
    return {
        "w": w,
        "history": pd.DataFrame(history),
        "nodes": pd.DataFrame(node_rows),
        "auc": history[-1]["test_auc"],
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": float((pred == actual).mean()),
        "threshold": thr,
        "n_test": int(len(yte)),
        "base_rate": float(actual.mean()),
    }


# ----------------------------------------------------------------------------
# RISK FUSION (per 1 km cell)
# ----------------------------------------------------------------------------
def level(r):
    if r < 0.25: return "🟢 Safe"
    if r < 0.45: return "🟡 Watch"
    if r < 0.65: return "🟠 Warning"
    return "🔴 Emergency"


def score_cells(grid, fc, clim, events, w, now):
    out = []
    for i, row in enumerate(grid.itertuples()):
        hf = fc[i]
        fut = hf[(hf.time >= now) & (hf.time < now + pd.Timedelta(hours=72))]
        p7 = hf[(hf.time < now) & (hf.time >= now - pd.Timedelta(days=7))]
        p24 = hf[(hf.time < now) & (hf.time >= now - pd.Timedelta(hours=24))]
        p72 = hf[(hf.time < now) & (hf.time >= now - pd.Timedelta(hours=72))]

        r72 = float(fut.precipitation.sum())
        a7 = float(p7.precipitation.sum())
        t72 = float(fut.temperature_2m.max()) if len(fut) else 0.0

        flood = np.clip(r72 / max(clim["p3_p95"], 20), 0, 1.5) / 1.5
        slide = np.clip((r72 + a7) / max(clim["p10_p95"], 30), 0, 1.5) / 1.5
        heat = np.clip((t72 - 36) / 8, 0, 1)                      # IMD heatwave band ≈ 40°C+
        p_ml = float(sig(feat_now(p24, p72, now.month) @ w))
        ml_s = min(p_ml / 0.25, 1.0)

        n_ev = 0
        if not events.empty:
            n_ev = sum(haversine(row.lat, row.lon, e.lat, e.lon) <= EVENT_RADIUS_KM
                       for e in events.itertuples())
        bonus = min(0.2, 0.05 * n_ev)

        risk = float(np.clip(0.35 * flood + 0.30 * slide + 0.25 * heat + 0.10 * ml_s + bonus, 0, 1))
        out.append({
            "cell": row.cell, "lat": row.lat, "lon": row.lon,
            "rain72_mm": round(r72, 1), "antecedent7d_mm": round(a7, 1),
            "tmax72_c": round(t72, 1), "ml_heavy_rain_prob": round(p_ml, 3),
            "events_near": n_ev, "risk": round(risk, 3), "level": level(risk),
        })
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛰️ ClimaGuard AI")
    st.caption("Hyperlocal early warning · live data")
    city = st.selectbox("City / region", list(CITIES.keys()) + ["Custom coordinates"], index=0)
    if city == "Custom coordinates":
        lat = st.number_input("Latitude", value=18.5204, format="%.4f")
        lon = st.number_input("Longitude", value=73.8567, format="%.4f")
    else:
        lat, lon = CITIES[city]
    k = st.slider("Grid radius (cells)", 1, 3, 2,
                  help="2 → 5×5 grid of ~1 km cells. Each cell gets its own live forecast.")
    if st.button("🔄 Refresh live data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.caption("Sources: Open-Meteo (forecast + ERA5 archive) • NASA EONET v3")

# ----------------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------------
now = datetime.now(IST).replace(tzinfo=None, minute=0, second=0, microsecond=0)
grid = make_grid(lat, lon, k)

try:
    with st.spinner("📡 Pulling live forecasts, 10-year climatology, NASA events and training the federated model…"):
        fc = fetch_forecast(tuple(grid.lat.round(4)), tuple(grid.lon.round(4)))
        arc = fetch_archive(round(lat, 4), round(lon, 4))
        events = fetch_events()
        fed = run_federated()
except Exception as e:
    st.error(f"Live data request failed: {e}. Check internet access and press **Refresh live data**.")
    st.stop()

clim = build_climatology(arc, now.month)
df = score_cells(grid, fc, clim, events, fed["w"], now)
center_idx = len(df) // 2

if not events.empty:
    events = events.copy()
    events["km_from_city"] = events.apply(lambda r: haversine(lat, lon, r.lat, r.lon), axis=1).round(0)
    near_events = int((events["km_from_city"] <= EVENT_RADIUS_KM).sum())
else:
    near_events = 0

top = df.loc[df["risk"].idxmax()]

# ----------------------------------------------------------------------------
# HEADER & KPIs
# ----------------------------------------------------------------------------
st.markdown(
    f'<div class="hero"><h1>🛰️ ClimaGuard AI</h1>'
    f'<p>Federated • multimodal • hyperlocal disaster early warning &nbsp;·&nbsp; '
    f'{city if city != "Custom coordinates" else f"{lat:.3f}, {lon:.3f}"} &nbsp;·&nbsp; '
    f'updated {now:%d %b %Y, %H:%M} IST</p></div>',
    unsafe_allow_html=True,
)


def kpi(label, value, sub, color):
    st.markdown(
        f'<div class="kpi" style="--c:{color}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi("Highest-risk cell", top["cell"], f"{top['level']} · score {top['risk']:.2f}", "#dc2626")
with k2:
    n_alert = int((df["risk"] >= 0.45).sum())
    kpi("Cells at Warning+", f"{n_alert} / {len(df)}", "of the 1 km grid", "#f97316")
with k3:
    kpi("72h rain (max cell)", f"{df['rain72_mm'].max():.1f} mm", "next 72 hours · live forecast", "#0ea5e9")
with k4:
    kpi("NASA events nearby", str(near_events), f"within {EVENT_RADIUS_KM} km · open EONET events", "#7c3aed")

st.write("")

tabs = st.tabs(["🗺️ Ward Risk Map", "⏱️ 72-Hour Outlook", "📈 Climatology",
                "🤝 Federated Model", "🚨 NASA Event Feed", "ℹ️ Data & Method"])

# ---- TAB 1: Ward risk map ---------------------------------------------------
with tabs[0]:
    c_map, c_tbl = st.columns([3, 2])
    with c_map:
        fig = px.scatter_geo(
            df, lat="lat", lon="lon", color="level", color_discrete_map=LEVEL_COLORS,
            hover_name="cell", hover_data=["risk", "rain72_mm", "tmax72_c", "events_near"],
            projection="mercator", height=540,
        )
        fig.update_traces(marker=dict(size=26, opacity=0.95))
        fig.update_geos(
            center={"lat": lat, "lon": lon},
            projection_scale=250,
            showland=True, landcolor="#263238",
            showocean=True, oceancolor="#101820",
            showcountries=True, countrycolor="#607d8b",
            showcoastlines=True, coastlinecolor="#90a4ae",
            bgcolor="#111111",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          paper_bgcolor="#111111",
                          legend=dict(title="Alert level", bgcolor="rgba(17,17,17,.9)",
                                      font=dict(color="#f8fafc")))
        st.plotly_chart(fig, use_container_width=True)
    with c_tbl:
        st.markdown("#### Ward-level risk ranking")
        show = df[["cell", "level", "risk", "rain72_mm", "tmax72_c", "events_near"]] \
            .sort_values("risk", ascending=False) \
            .rename(columns={"cell": "Cell (~1 km²)", "level": "Alert", "risk": "Risk",
                             "rain72_mm": "Rain 72h (mm)", "tmax72_c": "Max temp 72h (°C)",
                             "events_near": "NASA events"})
        st.dataframe(show, hide_index=True, use_container_width=True,
                     column_config={"Risk": st.column_config.ProgressColumn(
                         "Risk", min_value=0, max_value=1, format="%.2f")})
    st.caption("Risk = 0.35·flood + 0.30·landslide + 0.25·heat + 0.10·ML + NASA event bonus (≤0.2).")

# ---- TAB 2: 72-hour outlook -------------------------------------------------
with tabs[1]:
    cell_names = df["cell"].tolist()
    sel = st.selectbox("Choose a 1 km cell", cell_names, index=center_idx)
    hf = fc[cell_names.index(sel)]
    seg = hf[(hf.time >= now - pd.Timedelta(hours=24)) & (hf.time < now + pd.Timedelta(hours=72))]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=seg["time"], y=seg["precipitation"], name="Rain (mm/h)",
                marker_color="#38bdf8", secondary_y=False)
    fig.add_scatter(x=seg["time"], y=seg["temperature_2m"], name="Air temp (°C)",
                    line=dict(color="#f97316", width=3), secondary_y=True)
    fig.add_scatter(x=seg["time"], y=seg["apparent_temperature"], name="Feels like (°C)",
                    line=dict(color="#e11d48", dash="dot", width=2), secondary_y=True)
    fig.add_vline(x=now, line_dash="dash", line_color="#7c3aed")
    fig.update_layout(template="plotly_dark", height=460, title=f"Live hourly outlook · {sel}",
                      legend=dict(orientation="h", y=1.1),
                      margin=dict(l=10, r=10, t=60, b=10))
    fig.update_yaxes(title_text="Rain (mm/h)", secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    fut = hf[(hf.time >= now) & (hf.time < now + pd.Timedelta(hours=72))].copy()
    fut["Day"] = fut["time"].dt.strftime("%a %d %b")
    daily = fut.groupby("Day", sort=False).agg(
        **{"Max temp (°C)": ("temperature_2m", "max"),
           "Max feels-like (°C)": ("apparent_temperature", "max"),
           "Rain (mm)": ("precipitation", "sum")}).reset_index()
    st.markdown("#### Day-by-day 72-hour summary")
    st.dataframe(daily.round(1), hide_index=True, use_container_width=True)

# ---- TAB 3: Climatology -----------------------------------------------------
with tabs[2]:
    a = arc.copy()
    a["year"] = a["time"].dt.year
    yearly = a.groupby("year").agg(
        heavy_rain_days=("precipitation_sum", lambda s: int((s >= HEAVY_RAIN_MM).sum())),
        total_rain_mm=("precipitation_sum", "sum"),
        hottest_day_c=("temperature_2m_max", "max"),
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        f1 = px.bar(yearly, x="year", y="heavy_rain_days", title="Heavy-rain days per year (≥64.5 mm)",
                    color="heavy_rain_days", color_continuous_scale="Blues")
        f1.update_layout(template="plotly_dark", height=360, coloraxis_showscale=False)
        st.plotly_chart(f1, use_container_width=True)
    with c2:
        f2 = px.line(yearly, x="year", y="hottest_day_c", markers=True,
                     title="Hottest day per year (°C)", color_discrete_sequence=["#f97316"])
        f2.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(f2, use_container_width=True)

    a["month"] = a["time"].dt.month
    monthly = a.groupby("month").agg(
        avg_rain_mm=("precipitation_sum", "mean"),
        avg_tmax_c=("temperature_2m_max", "mean"),
        p90_tmax_c=("temperature_2m_max", lambda s: s.quantile(0.9)),
        p95_3day_rain_mm=("precipitation_sum", lambda s: s.rolling(3).sum().quantile(0.95)),
    ).reset_index()
    monthly["Month"] = monthly["month"].apply(lambda m: datetime(2000, m, 1).strftime("%b"))
    st.markdown(f"#### Monthly climatology · the thresholds used for today ({now:%B}) are the ones that matter")
    st.dataframe(monthly.drop(columns="month").round(1), hide_index=True, use_container_width=True)
    st.caption(f"Today's climatological thresholds → 3-day rain p95: {clim['p3_p95']:.1f} mm · "
               f"10-day rain p95: {clim['p10_p95']:.1f} mm · tmax p90: {clim['tmax_p90']:.1f} °C")

# ---- TAB 4: Federated model -------------------------------------------------
with tabs[3]:
    st.markdown("#### 🤝 Federated learning across 3 state disaster-management nodes")
    st.info("Each node trains on its own archive data. Only **model weights** travel to the central "
            "server, which averages them (FedAvg). Raw data never leaves the node.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Global ROC-AUC", f"{fed['auc']:.3f}")
    m2.metric(f"Precision @ {fed['threshold']}", f"{fed['precision']:.2f}")
    m3.metric("Recall", f"{fed['recall']:.2f}")
    m4.metric("Accuracy", f"{fed['accuracy']:.2f}")
    st.caption(f"Held-out test year: {fed['n_test']} days · base rate of heavy-rain days: "
               f"{fed['base_rate']*100:.1f}% · target = tomorrow ≥ {HEAVY_RAIN_MM} mm")

    c1, c2 = st.columns([3, 2])
    with c1:
        fh = px.line(fed["history"], x="round", y="test_auc", markers=True,
                     title="Global model AUC per federated round",
                     color_discrete_sequence=["#7c3aed"])
        fh.update_layout(template="plotly_dark", height=360, yaxis_range=[0, 1])
        st.plotly_chart(fh, use_container_width=True)
    with c2:
        st.markdown("#### State nodes")
        st.dataframe(fed["nodes"], hide_index=True, use_container_width=True)
        st.markdown("#### Global weights (shared)")
        st.dataframe(pd.DataFrame({"Feature": FEATS, "Weight": fed["w"].round(3)}),
                     hide_index=True, use_container_width=True)

    st.markdown("#### Live ML heavy-rain probability per cell (tomorrow)")
    st.bar_chart(df.set_index("cell")["ml_heavy_rain_prob"], color="#0ea5e9")

# ---- TAB 5: NASA event feed -------------------------------------------------
with tabs[4]:
    st.markdown("#### 🚨 Open NASA-observed natural events (EONET)")
    if events.empty:
        st.success("No open floods, landslides, severe storms or temperature extremes in the EONET feed right now.")
    else:
        ev_show = events.sort_values("km_from_city")[["title", "category", "date", "km_from_city", "link"]] \
            .rename(columns={"title": "Event", "category": "Type", "date": "Date",
                             "km_from_city": "km from city", "link": "Source"})
        st.dataframe(ev_show, hide_index=True, use_container_width=True,
                     column_config={"Source": st.column_config.LinkColumn("Source")})
        fe = px.scatter_map(events, lat="lat", lon="lon", color="category", hover_name="title",
                            zoom=3, height=480, map_style="open-street-map")
        fe.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fe, use_container_width=True)

# ---- TAB 6: Data & method ---------------------------------------------------
with tabs[5]:
    st.markdown("""
#### How this maps to the ClimaGuard AI deck

| Deck layer | What this app actually does | Status |
|---|---|---|
| Satellite (NASA / ESA) | NASA EONET open events (satellite-observed floods, storms, landslides, heat) | ✅ Live |
| IMD historical | Open-Meteo ERA5 archive for the same coordinates (IMD gridded data needs manual download) | ✅ Live, substitute |
| Forecast / lead time | Open-Meteo hourly forecast, 72 h, per ~1 km cell | ✅ Live |
| Federated learning | FedAvg logistic model across 3 state nodes, held-out year evaluation | ✅ Live training |
| Ward-level risk | 5×5 (or 7×7) grid of ~1 km cells, each with its own forecast | ✅ Live |

**Flood score:** 72h forecast rain vs. this month's 95th-percentile 3-day rainfall.
**Landslide score:** 72h forecast rain + previous 7 days vs. the 95th-percentile 10-day total.
**Heat score:** 72h max temperature, scaled from 36 °C (0) to 44 °C (1).
**ML score:** federated model's probability of ≥64.5 mm rain tomorrow, scaled so 25% = maximum.
**Alert levels:** Safe < 0.25 · Watch < 0.45 · Warning < 0.65 · Emergency ≥ 0.65.

> **Prototype disclaimer:** this is a decision-support prototype, not an official warning system. Official alerts come from IMD and state authorities. Model metrics above are measured on one held-out year, not a full validation study.
""")