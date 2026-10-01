# 🌊 OceanTrace (Ocean-Trace-v3.0)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-U--Net-EE4C2C?logo=pytorch&logoColor=white) ![Vite](https://img.shields.io/badge/Vite-Frontend-646CFF?logo=vite&logoColor=white) ![OpenDrift](https://img.shields.io/badge/OpenDrift-Ocean%20Physics-0077B6) ![NetworkX](https://img.shields.io/badge/NetworkX-Routing-2E8B57) ![License](https://img.shields.io/badge/License-MIT-green)

**Unified Satellite SAR Hazard Intelligence: Oil Spill Detection & Vessel Attribution + Antarctic Iceberg Trajectory & Navigation Decision Support**

*Smart India Hackathon 2026 — Problem Statement SIH26059, Ministry of Earth Sciences (MoES) / National Centre for Polar and Ocean Research (NCPOR)*

---

## 🧭 About the Project

OceanTrace is a full-stack maritime hazard intelligence platform built around one core idea: a satellite radar (SAR) image doesn't come pre-labeled with what hazard it's showing you. It detects marine oil spills and Antarctic icebergs from the same Sentinel-1 SAR pipeline, then routes each to the response it actually needs — backward physics reconstruction and AIS vessel attribution for oil spills, forward drift forecasting and safe-route planning for icebergs.

**The problem it solves:** When a vessel illegally discharges oil or suffers an accident at sea, the slick drifts with currents and wind — by the time a satellite captures it, the responsible ship may be hundreds of kilometers away. Separately, research vessels navigating Antarctic waters rely on infrequent, low-resolution ice charts instead of a live, ship-specific predictive hazard view. Both are, at their core, the same unsolved problem: **a radar anomaly is detected, and nobody turns that detection into a fast, actionable, evidence-backed decision.**

OceanTrace automates both chains — detect → classify → physically model → respond — through a physics-informed, evidence-based pipeline that produces transparent, auditable output rather than an opaque black-box verdict.
                🛰️ Sentinel-1 SAR Image
                          │
                          ▼
    ┌──────────────────────────────────────────┐
    │ MODULE 1 — Unified Hazard Detection        │
    │ • ResNet-34 U-Net (PyTorch, pretrained)    │ ──► detection.json
    │ • Hazard classifier: Oil / Iceberg / None  │     (Polygon, Area, Confidence, hazard_type)
    │ • Look-alike false-positive filtering      │
    └──────────────────────────────────────────┘
                          │
          ┌───────────────┴────────────────┐
          ▼ hazard_type = "oil"             ▼ hazard_type = "iceberg"

┌──────────────────────────────┐ ┌──────────────────────────────────┐
│ MODULE 2 — Backward Drift │ │ MODULE 2 — Forward Drift │
│ Reconstruction (OpenDrift) │ │ Forecast (OpenDrift) │
│ • Real CMEMS current + ERA5 │ │ • 24–72h predicted position │
│ • Reverse-time simulation │ │ • 68%/95% uncertainty ellipses │
└──────────────────────────────┘ └──────────────────────────────────┘
│ │
▼ ▼
┌──────────────────────────────┐ ┌──────────────────────────────────┐
│ MODULE 3 — AIS Vessel │ │ MODULE 3 — Route-Risk Scoring │
│ Attribution │ │ • Hazard-proximity + ellipse │
│ • 5-factor evidence scoring │ │ breach detection │
└──────────────────────────────┘ └──────────────────────────────────┘
│ │
▼ ▼
📄 Forensic PDF Report ┌──────────────────────────────────┐
(SHA-256 sealed, QR-verifiable) │ MODULE 4 — Safe-Path Planning │
│ │ • NetworkX graph routing around │
▼ │ forecasted hazard zones │
🗄️ PostgreSQL Incident & Vessel └──────────────────────────────────┘
Registry │
▼
Recommended safest navigation route


---

## 🤔 Why Oil Spills and Icebergs, in One Platform

This isn't two hackathon projects stitched together — it's a deliberate decision grounded in a real technical fact.

- **Shared false-positive problem, solved once.** In SAR literature, icebergs are a documented "look-alike" for oil slicks — both dampen radar backscatter similarly. A detector that already has to filter iceberg-like noise out of oil detection is one retraining step away from recognizing icebergs as a hazard in their own right, instead of discarding that signal.
- **Shared physics engine, used as designed.** OpenDrift models the trajectory of any floating object — oil, icebergs, debris — not oil alone. Running it backward for spill-origin hindcasting and forward for iceberg-drift forecasting is its intended dual use.
- **Shared operational need.** Coast guards and polar agencies don't want five disconnected single-purpose tools watching the same ocean — they want one feed, one pipeline, the correct response triggered automatically.
- **A converging real-world risk.** As polar ice retreats and new shipping lanes open, the same vessels face both oil-spill risk and iceberg risk in the same waters. A tool built for only one hazard ages badly.

> **The human impact:** a ship's crew near Antarctica gets a live, ship-specific warning instead of a weekly ice chart. A coastal community hit by a spill gets a traceable, accountable vessel instead of a slick that's seen and forgotten. Researchers and policymakers get one explainable evidence trail instead of a black box.

---

## 🎯 Target Users

| User | Iceberg Use Case | Oil Spill Use Case |
|---|---|---|
| **Ministry of Earth Sciences / NCPOR / Indian Antarctic Programme** | Live sea-ice and iceberg hazard tracking with drift forecasts and safe-route recommendations for Bharati & Maitri resupply voyages | — |
| **Research Vessel Captains & Expedition Planners** | Voyage-specific, explainable risk scoring and safe-path planning around forecasted iceberg zones | — |
| **Coast Guards & Maritime Surveillance Authorities** (ICG / USCG / EMSA) | Extendable to polar/sub-polar surveillance as ice-covered routes expand | Rapidly detect offshore pollution events and generate forensic-grade evidence dossiers |
| **Environmental Protection Agencies / Pollution Control Boards** | — | Hold polluters accountable for illegal bilge dumps and quantify environmental damage |
| **Port & Harbor Authorities** | — | Monitor near-shore anchorage zones and verify vessel compliance |
| **Maritime & Polar Researchers** | Iceberg drift data with direct value for glaciology research | Reproducible architecture combining Earth Observation data with Lagrangian ocean-physics modeling |

---

## ✨ Key Features

🛰️ **Deep Learning SAR Segmentation** — A ResNet-34 U-Net (transfer-learned from ImageNet, fine-tuned on real Sentinel-1 SAR imagery) trained with a custom Hybrid Focal Tversky loss to handle extreme class imbalance and isolate true hazards from look-alikes (calm-water zones, algal blooms, coastal wetlands).

🧠 **Hazard-Type Classification** — A dedicated classifier, trained on real labeled SAR data (Statoil/C-CORE iceberg dataset), tags each detected anomaly as Oil Slick, Iceberg, or False Alarm — routing it to the correct downstream pipeline.

🔄 **Unified Drift Engine** — OpenDrift-based Lagrangian particle simulation, run **backward** (real CMEMS current + ERA5 wind) to reconstruct an oil spill's origin, or **forward** to forecast an iceberg's drift 24–72 hours ahead with 68%/95% confidence ellipses.

🚢 **Transparent Multi-Factor Vessel Attribution** — Evaluates each candidate vessel across five independently-weighted, explainable signals (spatial proximity, temporal alignment, trajectory consistency, drift-path match, speed/course anomaly) instead of an opaque black-box score.

🧭 **Iceberg Route-Risk Scoring & Safe-Path Planning** — Scores a ship's planned route against forecasted iceberg hazard zones (Low/Medium/High, with plain-language explanation), and a NetworkX-based pathfinding engine recommends the lowest-risk route around multiple icebergs.

💻 **Interactive Full-Stack Web Console** — A modern geospatial dashboard with an interactive map, slick/iceberg overlay visualization, drift particle rendering, ranked vessel trajectories or recommended routes, live analytics, and printable/exportable forensic reports.

🔐 **Court-Ready Forensic PDF Export** — Client-side generated incident report with a SHA-256 tamper-evidence hash and a QR-verifiable case seal — no extra backend infrastructure required.

🗄️ **Persistent Incident Database** — Every detection, attribution, and route-risk report is stored in PostgreSQL, with automatic vessel-ownership enrichment via a maritime registry lookup.

🔌 **Standardized REST API** — Complete FastAPI service with CORS support, dual JSON/file-upload input, and interactive OpenAPI documentation (`/docs`).

---

## 🏗️ Architecture & Design Principles

- **Additive, non-destructive extension** — The iceberg capability was built entirely as new files and new branches alongside the original oil-spill modules; no existing oil-spill model weights, training pipelines, or attribution logic were modified, and all original test suites pass unchanged.
- **Modular, contract-driven pipeline** — Each module is independently developed, tested, and owned, with a formally versioned JSON schema (`contracts/CONTRACTS.md`) defining the handoff between them. A single `hazard_type` field determines which downstream path runs.
- **Real data, not synthetic** — Oil-spill model trained on a published Sentinel-1 SAR benchmark and validated against a real post-Hurricane-Ida Gulf of Mexico scene; iceberg classifier trained on the real Statoil/C-CORE SAR-labeled dataset; attribution tested against real NOAA MarineCadastre AIS records (264,646 points, 396 vessels).
- **Explainability over black-box confidence** — Both vessel attribution and route-risk scoring use transparent, weighted evidence rather than an opaque ML classifier, so every result can be justified signal-by-signal.
- **Honest engineering** — False-positive suppression (5 km coastal exclusion buffer + minimum polygon area filtering) was validated by reducing a real scene from 69,176 km² of false positives down to a confirmed, honest zero open-water detections.

---

## 🔒 Security

- **No secrets in source control** — credentials and API keys are read from environment variables (`.env`, excluded via `.gitignore`); only `.env.example` templates with placeholder values are committed.
- **Parameterized database queries** — all PostgreSQL access uses parameterized statements, not string-formatted SQL, to prevent injection.
- **Input validation at the API boundary** — FastAPI endpoints use Pydantic models to validate incoming request data before it reaches any detection, drift, attribution, or routing logic.
- **File upload validation** — uploaded SAR imagery is checked for valid format/content before processing.
- **CORS scoped for deployment** — restricted to the deployed frontend origin in production rather than left fully open.
- **Tamper-evident reporting** — forensic PDF exports are sealed with a client-side SHA-256 hash and QR-verifiable case ID.

---

## 🚀 Setup & Run Guide

Follow these exact steps to run the complete project on Windows, macOS, or Linux.

### 📋 Prerequisites
- Git ([Download](https://git-scm.com/downloads))
- Python 3.10+ ([Download](https://www.python.org/downloads/))
- Node.js 18+ & npm ([Download](https://nodejs.org/))
- PostgreSQL (optional — required only for persistent incident storage)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/senvidit4-alt/Ocean-Trace-v2.0.git
cd Ocean-Trace-v2.0
```

### Step 2 — Backend Setup (FastAPI & AI Pipeline)

**Windows (PowerShell):**
```powershell
# 1. Create Python virtual environment
python -m venv venv

# 2. Allow script execution (if PowerShell restricts scripts)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Install all Python dependencies
pip install -r requirements.txt

# 5. Start the backend server
python main.py
```

**Linux / macOS:**
```bash
# 1. Create Python virtual environment
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install all Python dependencies
pip install -r requirements.txt

# 4. Start the backend server
python3 main.py
```

✅ Backend live at: `http://localhost:8000`
📖 Interactive Swagger Docs: `http://localhost:8000/docs`
🩺 Health Check: `http://localhost:8000/api/health`

### Step 3 — Frontend Setup (Interactive Web Console)

Open a new terminal window:
```bash
cd ocean-trace-frontend
npm install
npm run dev
```
🖥️ Web Console live at: `http://localhost:5173`

### Step 4 — (Optional) Database Setup

```bash
cp .env.example .env
# Fill in your PostgreSQL credentials in .env
python report_db.py
# Initializes the schema
```

---

## 🧪 Automated Verification & Testing

Run the full integration test suite (from repo root, with `venv` activated):
```bash
python test_backend_api.py
```
Expected output:
============================================================
ALL API ENDPOINT INTEGRATION TESTS PASSED SUCCESSFULLY!

Run the full pipeline end-to-end via CLI (oil-spill path by default):
```bash
python run_pipeline.py --input-synthetic --output-dir outputs
```

Run the iceberg path:
```bash
python run_pipeline.py --hazard-type iceberg --output-dir outputs
```

---

## 📁 Repository Structure

Ocean-Trace-v2.0/
├── backend/
│ ├── init.py
│ └── main.py # FastAPI REST API endpoints — oil-spill + iceberg routes
├── ocean-trace-frontend/ # Interactive Web UI Console
│ ├── index.html # Main UI layout & geospatial map
│ ├── src/styles.css # Glassmorphism & dark-mode styling
│ ├── vite.config.js # Vite server & proxy configuration
│ ├── .env.example # Frontend environment variable template
│ └── package.json
├── modules/
│ ├── 01_detection/
│ │ └── src/
│ │ ├── model.py # Oil-spill ResNet-34 U-Net
│ │ ├── data_loader.py
│ │ ├── train.py
│ │ └── iceberg_classifier.py # Hazard-type classifier
│ ├── 02_drift/
│ │ └── src/
│ │ ├── source_reconstruction.py # Oil backward hindcast
│ │ └── iceberg_forecast.py # Iceberg forward forecast
│ ├── 03_attribution/
│ │ └── src/
│ │ ├── evidence_fusion.py # Oil vessel attribution
│ │ └── route_risk.py # Iceberg route-risk scoring
│ └── 04_navigation/ # Iceberg navigation module
│ └── src/
│ └── safe_path.py # NetworkX safe-path planning
├── data/
│ ├── AIS_*.csv # Real Gulf of Mexico AIS telemetry dataset
│ └── iceberg_dataset/
│ └── train.json # Statoil/C-CORE iceberg SAR training data
├── contracts/ # JSON schemas defining inter-module handoffs
├── docs/ # Architecture specifications & API documentation
├── unet_spill_best.pth # Pre-trained ResNet-34 U-Net weights
├── iceberg_classifier_best.pth # Trained iceberg hazard classifier
├── report_db.py # PostgreSQL incident/vessel storage interface
├── run_pipeline.py # Full CLI pipeline runner — branches by hazard_type
├── test_backend_api.py # Integration test suite for backend API
├── requirements.txt # Production Python dependencies
├── .env.example # Backend environment configuration template
└── main.py # Root entrypoint (python main.py)


---

## ⚙️ REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` or `/api/health` | Service health status & connected module readiness |
| `POST` | `/detect-spill` | Module 1 — Upload SAR TIFF, get hazard polygon, type, area & confidence |
| `POST` | `/trace-origin` | Module 2 (oil) — Run backward drift simulation from a detection polygon |
| `POST` | `/attribute-vessel` | Module 3 (oil) — Match candidate vessels against the source trajectory |
| `POST` | `/score-route-risk` | Module 3 (iceberg) — Score a planned route against a forecasted iceberg hazard |
| `POST` | `/find-safe-path` | Module 4 (iceberg) — Recommend the safest route around forecasted hazards |
| `POST` | `/run-full-pipeline` | Full end-to-end pipeline execution, auto-branching by detected hazard type |

Full interactive API reference available at `/docs` once the backend is running.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Detection** | PyTorch, `segmentation_models_pytorch` (ResNet-34 U-Net), hazard classifier CNN, rasterio, OpenCV |
| **Drift & Forecasting** | OpenDrift (backward hindcast + forward forecast), NetCDF, CMEMS / ECMWF ocean-current & wind data |
| **Oil Vessel Attribution** | pandas, GeoPandas, Shapely, NOAA MarineCadastre AIS data |
| **Iceberg Route Risk & Navigation** | GeoPandas, Shapely, NetworkX (graph-based safe-path routing) |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Database** | PostgreSQL, psycopg2, PostGIS, JSONB |
| **Frontend** | Vite, HTML/CSS/JavaScript, Leaflet-style map rendering |
| **Forensic Export** | html2pdf.js, Web Crypto API (SHA-256), qrcode.js |
| **Deployment** | Render (backend), Vercel (frontend) |

---

## 🛠️ Troubleshooting

- **PowerShell execution policy error (Windows)** — Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating the virtual environment.
- **Port 8000 or 5173 already in use** — Run the backend on a different port: `uvicorn main:app --port 8080`, and update `VITE_API_URL` in `ocean-trace-frontend/.env` accordingly.
- **Pretrained weights check** — `unet_spill_best.pth` and `iceberg_classifier_best.pth` are already included in the repository root; the backend automatically detects and loads both on startup.
- **Database connection errors** — Ensure PostgreSQL is running locally and credentials in `.env` match your local setup; the app runs fully without a database, just without persistent storage.

---

## 🔮 Future Scope

- Live global vessel traffic integration via real-time AIS streaming (e.g. AISstream.io) for continuous maritime monitoring beyond historical incident review.
- Extension of the hazard classifier to additional SAR look-alike categories (algal blooms, biogenic films) for even lower false-positive rates.
- Deployment against live Antarctic and Indian Ocean satellite coverage as a real-time regional monitoring roadmap.

---

## 👥 Team

**Team Ocean Trace** — Smart India Hackathon 2026
**Problem Statement:** SIH26059 (Ministry of Earth Sciences / NCPOR)
**Repository:** [github.com/senvidit4-alt/Ocean-Trace-v2.0](https://github.com/senvidit4-alt/Ocean-Trace-v2.0)

## 📄 License

This project is licensed under the MIT License.
