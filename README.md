# 🛰️ ClimaGuard AI

**Federated, multimodal, hyperlocal disaster early-warning dashboard**

ClimaGuard AI is a single-file [Streamlit](https://streamlit.io) application that turns live public climate data into a ward-level (~1 km²) risk map for floods, heatwaves and landslides. It forecasts 72 hours ahead, fuses several signals into one interpretable risk score per grid cell, and demonstrates a privacy-preserving **federated learning** setup across three simulated state disaster-management nodes.

> **Prototype notice:** ClimaGuard AI is a decision-support prototype. It is **not** an official warning system. Official alerts come from the India Meteorological Department (IMD) and state disaster-management authorities.

---

## Table of Contents

- [Why ClimaGuard AI](#why-climaguard-ai)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Data Sources](#data-sources)
- [Risk Model](#risk-model)
- [Federated Learning](#federated-learning)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Why ClimaGuard AI

District-level forecasts cover roughly 8,000 km² on average, which is too coarse for local action. Two communities in the same district can face very different flood or heat risk. ClimaGuard AI narrows that gap by:

- Evaluating risk on a **~1 km grid** around any chosen location
- Giving each cell its **own 72-hour forecast**
- Combining **climatological thresholds, forecast extremes, recent rainfall, satellite-observed events and an ML model** into a single score
- Training the model **without pooling raw data** from different administrative nodes

## Key Features

| Feature | Description |
|---|---|
| **Ward-level risk map** | Interactive map of every grid cell, colour-coded by alert level |
| **72-hour outlook** | Hourly rain and temperature chart with a day-by-day summary |
| **Climatology** | Ten years of ERA5 reanalysis: heavy-rain days, hottest days, monthly thresholds |
| **Federated model** | FedAvg logistic regression across three nodes, with per-round AUC |
| **NASA event feed** | Open floods, landslides, severe storms and temperature extremes from NASA EONET |
| **No API keys** | All data sources are free and public |
| **Single file** | The entire application lives in one Python script |

## Architecture

ClimaGuard AI follows a four-layer pipeline: **observe → analyse → learn privately → alert**.

```
┌───────────────────────┐   ┌──────────────────────────┐   ┌────────────────────────┐   ┌─────────────────────┐
│ 1. DATA LAYER         │ → │ 2. AI / ML LAYER         │ → │ 3. FEDERATED LAYER     │ → │ 4. OUTPUT LAYER     │
│                       │   │                          │   │                        │   │                     │
│ Open-Meteo forecast   │   │ Hourly feature extraction│   │ State node 1 (local)   │   │ Ward risk map       │
│ Open-Meteo ERA5 archive│  │ Climatology thresholds   │   │ State node 2 (local)   │   │ 72h forecast chart  │
│ NASA EONET events     │   │ Risk fusion (per cell)   │   │ State node 3 (local)   │   │ Alert levels        │
│                       │   │ Logistic model (FedAvg)  │   │ → central weights only │   │ Event feed          │
└───────────────────────┘   └──────────────────────────┘   └────────────────────────┘   └─────────────────────┘
```

**Privacy boundary:** raw data never leaves its owning node. The central aggregator only receives model weights.

## Data Sources

| Source | Used for | Access |
|---|---|---|
| [Open-Meteo Forecast API](https://open-meteo.com/en/docs) | Hourly 72-hour forecast and 7-day antecedent rainfall for each cell | Free, no key |
| [Open-Meteo Archive API (ERA5)](https://open-meteo.com/en/docs/historical-weather-api) | Daily rainfall and maximum temperature, 2016 to present | Free, no key |
| [NASA EONET v3](https://eonet.gsfc.nasa.gov/docs/v3) | Open satellite-observed natural events | Free, no key |

Forecasts are cached for 30 minutes, archive data for 24 hours, and the federated training run for 24 hours.

## Risk Model

Each 1 km cell receives a risk score between 0 and 1:

```
risk = 0.35 × flood + 0.30 × landslide + 0.25 × heat + 0.10 × ML + event_bonus
```

| Component | Calculation |
|---|---|
| **Flood** | 72-hour forecast rainfall ÷ this month's 95th-percentile 3-day rainfall |
| **Landslide** | (72-hour forecast rainfall + previous 7 days) ÷ 95th-percentile 10-day rainfall |
| **Heat** | 72-hour maximum temperature, scaled from 36 °C (0) to 44 °C (1) |
| **ML** | Federated model's probability of ≥ 64.5 mm rain tomorrow, scaled so 25 % = 1.0 |
| **Event bonus** | +0.05 per open NASA event within 250 km, capped at +0.20 |

**Alert levels**

| Level | Score |
|---|---|
| 🟢 Safe | < 0.25 |
| 🟡 Watch | 0.25 – 0.45 |
| 🟠 Warning | 0.45 – 0.65 |
| 🔴 Emergency | ≥ 0.65 |

The 64.5 mm threshold follows IMD's definition of heavy rainfall.

## Federated Learning

The model is a logistic regression trained with **Federated Averaging (FedAvg)**:

1. Each of three simulated state nodes (Maharashtra, Gujarat, Karnataka) downloads its own archive data.
2. Every round, the server sends the current global weights to each node.
3. Each node trains locally for 200 gradient steps and returns only its updated weights.
4. The server averages the weights, weighted by each node's sample count.
5. After six rounds, the global model is evaluated on a held-out final year.

**Features:** today's rainfall, 3-day rainfall, maximum temperature, seasonal sine and cosine terms, and a bias term.

**Target:** whether tomorrow's rainfall reaches 64.5 mm.

Reported metrics include ROC-AUC, precision, recall and accuracy on the held-out year. These are measured on one year of data and should not be read as a full validation study.

## Getting Started

### Prerequisites

- Python 3.9 or later
- Internet access (the app pulls live data on every cold start)
- Plotly **5.24 or later**, which provides `px.scatter_map`

### Installation

```bash
# 1. Clone or copy the project
git clone https://github.com/<your-username>/climaguard-ai.git
cd climaguard-ai

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install streamlit plotly pandas numpy requests
```

### Run

```bash
streamlit run climaguard_app.py
```

The dashboard opens at `http://localhost:8501`. The first load takes roughly 10–30 seconds while the app fetches and trains.

## Usage

1. **Choose a location** from the sidebar, or select *Custom coordinates* and enter latitude and longitude.
2. **Set the grid radius.** A value of 2 produces a 5 × 5 grid (about 25 km²). Values up to 3 produce a 7 × 7 grid.
3. **Review the tabs:**
   - **Ward Risk Map**: spatial overview and ranked table
   - **72-Hour Outlook**: hourly forecast for any selected cell
   - **Climatology**: historical context and current thresholds
   - **Federated Model**: training curve, node statistics and live ML probabilities
   - **NASA Event Feed**: nearby open events with source links
   - **Data & Method**: full methodology and data-status table
4. **Click 🔄 Refresh live data** to clear the cache and re-fetch everything.

## Configuration

Most settings are constants near the top of `climaguard_app.py`:

| Constant | Default | Purpose |
|---|---|---|
| `HEAVY_RAIN_MM` | `64.5` | Heavy-rain threshold for labels and the ML target |
| `EVENT_RADIUS_KM` | `250` | Search radius for NASA events |
| `CITIES` | 9 cities | Preset locations in the sidebar |
| `NODES` | 3 nodes | Simulated federated state nodes |
| `LEVEL_COLORS` | — | Colours for each alert level |
| `run_federated(rounds=6)` | `6` | Number of federated training rounds |
| `thr` (inside `run_federated`) | `0.15` | Probability threshold for precision and recall |

To add a city, add an entry to `CITIES`:

```python
CITIES["Nagpur, Maharashtra"] = (21.1458, 79.0882)
```

## Project Structure

```
climaguard-ai/
├── climaguard_app.py   # Entire application (data, model, UI)
└── README.md           # This file
```

## Limitations

- **Not an official warning system.** Use IMD and state authority alerts for decisions.
- **Reanalysis substitutes for IMD gridded data.** The app uses ERA5 through Open-Meteo. IMD gridded data requires a manual download.
- **EONET is not raw satellite processing.** It aggregates events that have been detected by satellite and other sources.
- **Model evaluation is limited.** Metrics come from one held-out year. Cross-district and multi-year validation is still needed.
- **Heuristic weights.** The risk-fusion weights (0.35, 0.30, 0.25, 0.10) are chosen by hand, not fitted to outcomes.
- **Forecast uncertainty.** Forecasts beyond a few days carry substantial error, and the app does not show uncertainty bands.

## Roadmap

- [ ] Add IMD gridded rainfall as an input once a download pipeline is in place
- [ ] Replace the logistic model with an LSTM or Transformer for time-series prediction
- [ ] Add Hindi and regional-language UI
- [ ] Support multi-year, cross-district validation with MAE and RMSE reporting
- [ ] Export alerts as CAP (Common Alerting Protocol) messages
- [ ] Package the federated layer with Flower (`flwr`) for real distributed training

## Troubleshooting

| Problem | Fix |
|---|---|
| `AttributeError` on `px.scatter_map` | Upgrade Plotly: `pip install -U "plotly>=5.24"` |
| Blank page or "Live data request failed" | Check internet access, then click **Refresh live data** |
| Slow first load | Expected. Results are cached afterwards |
| Open-Meteo rate-limit error | Wait a minute, reduce the grid radius, or refresh less often |
| `use_container_width` deprecation warning | Harmless on recent Streamlit. Replace with `width="stretch"` if your version requires it |

## Acknowledgements

- [Open-Meteo](https://open-meteo.com) for free forecast and ERA5 archive access
- [NASA EONET](https://eonet.gsfc.nasa.gov) for open natural-event data
- [Streamlit](https://streamlit.io) and [Plotly](https://plotly.com) for the interface and charts
- India Meteorological Department (IMD) for the heavy-rainfall definition used here

## License

Specify your license here, for example MIT. Add a `LICENSE` file to the repository.

---

<p align="center">
  <b>ClimaGuard AI</b> · Practical climate early warning, built for local action<br>
  <sub>PCCOE International Grand Challenge 2026 · Theme: AI for Climate Change</sub>
</p>
