🌊 OceanTrace — Unified Maritime Hazard Intelligence Platform

AI-Powered Detection, Physics-Based Drift Forecasting & Response for Oil Spills and Antarctic Icebergs



Built for Smart India Hackathon 2026 Problem Statement: SIH26059 — AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System Organization: Ministry of Earth Sciences (MoES) · Department: National Centre for Polar and Ocean Research (NCPOR) · Category: Software · Theme: Transportation & Logistics

🧭 About the Project

OceanTrace is a unified, satellite-driven maritime hazard intelligence platform. It was originally built as a dedicated oil-spill detection and vessel-attribution system, and has since been extended into a broader multi-hazard ocean monitoring platform that also detects Antarctic icebergs, forecasts their drift, and recommends safe navigation routes for research vessels — all from the same core satellite radar (SAR) pipeline.

The core insight behind combining these two problems: Oil slicks and icebergs are both radar-detectable ocean anomalies that are notoriously easy to confuse with each other in SAR imagery — both dampen radar backscatter and appear as similar dark/anomalous patches. Any robust SAR-based hazard detector already has to solve the problem of telling them apart. Rather than building two isolated, single-purpose tools, OceanTrace turns this shared discrimination problem into a single platform: one satellite feed, one detection pipeline, two hazard responses.

                    🛰️  Sentinel-1 SAR Image (any pass, any region)
                                 │
                                 ▼
        ┌─────────────────────────────────────────────────────┐
        │  MODULE 1 — Unified Hazard Detection                  │
        │  • ResNet-34 U-Net (PyTorch) — anomaly segmentation   │
        │  • Hazard classifier — Oil / Iceberg / False Alarm    │
        └─────────────────────────────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼ hazard_type = "oil"            ▼ hazard_type = "iceberg"
   ┌───────────────────────────┐     ┌───────────────────────────────────┐
   │ MODULE 2 — Backward         │     │ MODULE 2 — Forward                  │
   │ Drift Hindcast (OpenDrift)  │     │ Drift Forecast (OpenDrift)          │
   │ → Spill origin & time       │     │ → 24–72h predicted iceberg position │
   └───────────────────────────┘     └───────────────────────────────────┘
                 │                                 │
                 ▼                                 ▼
   ┌───────────────────────────┐     ┌───────────────────────────────────┐
   │ MODULE 3 — AIS Vessel       │     │ MODULE 3 — Route-Risk Scoring       │
   │ Attribution (5-factor       │     │ (hazard-proximity, uncertainty       │
   │ transparent evidence score) │     │ ellipse breach, risk verdict)        │
   └───────────────────────────┘     └───────────────────────────────────┘
                 │                                 │
                 ▼                                 ▼
    Ranked suspect vessel list      ┌───────────────────────────────────┐
    + forensic PDF report            │ MODULE 4 — Safe-Path Planning (NEW) │
                                      │ NetworkX graph routing around       │
                                      │ forecasted iceberg hazard zones     │
                                      └───────────────────────────────────┘
                                                   │
                                                   ▼
                                     Recommended safest navigation route + Pdf report for future analysis 
**🤔 Why We Combined Two Problem Statements Into One Platform**

This isn't two projects stitched together for convenience — it's a deliberate architectural decision grounded in a real technical fact:

Shared false-positive problem, solved once. In SAR-based oil-spill detection literature, icebergs are explicitly documented as one of the classic "look-alikes" that cause false positives — both oil slicks and icebergs dampen radar backscatter similarly. Our original oil-spill detector already had to learn to filter out iceberg-like signatures as noise. We formalized that same discrimination boundary into a full second hazard class, turning a filter into a detector.
Shared physics engine, used as designed. OpenDrift — the Lagrangian ocean-physics engine we use for drift modeling — is explicitly built to model the trajectory of any floating object (oil, icebergs, debris, search-and-rescue targets), not oil alone. Running it backward for oil-origin hindcasting and forward for iceberg-drift forecasting is the engine's intended dual use, not a workaround.
Shared operational need. Coast guards, polar research agencies, and port authorities don't want five disconnected single-purpose tools watching the same ocean. A unified maritime hazard awareness platform — one satellite feed, one pipeline, automatic routing to the correct response — mirrors how real maritime surveillance centers are expected to operate.
A growing real-world overlap. As Arctic and Antarctic sea routes open up due to retreating ice, the same vessels navigating increasingly ice-exposed waters are also the ones at risk of causing or encountering oil spills in environmentally sensitive regions — the two hazards are converging in the same operational theatre, not just in our codebase.

In short: detect any SAR anomaly → classify what it actually is → predict where it's going → trigger the correct response (who's responsible, for oil; how to navigate safely, for ice).

💡 Why This Combination Actually Matters

It's easy to assume we combined these two problems just to save time. The honest truth is simpler, and a little more interesting: the ocean doesn't separate its hazards into neat categories, and neither should the systems that watch it.

A satellite doesn't know, ahead of time, whether the dark patch it just photographed is oil or ice. A ship's crew doesn't get to choose which hazard they'll encounter on a given voyage. The same radar pass over the Southern Ocean could just as easily reveal a drifting iceberg as it could an illegal discharge near a fishing fleet. Treating these as two unrelated problems, each needing its own from-scratch system, ignores how similar the first, hardest step — "what is this thing I'm looking at?" — really is for both.

What we gained by combining them:

A system that's harder to fool, not just twice as big. Teaching one model to tell oil apart from ice apart from ordinary calm water makes it better at all three distinctions at once — the kind of robustness that's genuinely difficult to achieve when you only ever show a model one hazard type.
Less wasted engineering, more depth where it counts. Instead of splitting our limited hackathon time across two completely separate codebases, we reused a proven detection-and-drift foundation and spent the time we saved on the parts that are genuinely new and hard — the iceberg route-risk scoring and the safe-path planning engine.
A platform that scales with the problem, not around it. Climate change is quietly merging these two risks in the real world — as polar ice retreats and new shipping lanes open, more vessels will be operating in waters where both oil-spill risk and iceberg risk coexist. A tool built for only one of them will age badly. A tool built to recognize any ocean hazard and route the right response won't.

The human impact, plainly stated:

For a ship's crew navigating near Antarctica, this is the difference between relying on a weekly ice chart and having a live, forecasted, ship-specific warning before they're ever close enough to be in danger.
For a coastal community whose fishing grounds get hit by an oil spill, this is the difference between a slick that's seen and forgotten, and one that's traced back to an actual, accountable vessel.
For polar researchers and policymakers, it's a single source of evidence — detection, physics, and explainable reasoning — that doesn't ask them to trust a black box.

We didn't build two projects and staple them together. We built one answer to a question that's bigger than either hazard alone: how do we give the people watching our oceans a system that sees what's actually out there, and tells them, honestly and clearly, what to do about it?

🎯 Target Users
User	Iceberg Capability	Oil Spill Capability
Ministry of Earth Sciences / NCPOR / Indian Antarctic Programme	Live sea-ice and iceberg hazard tracking with drift forecasts and safe-route recommendations for Bharati & Maitri resupply voyages	—
Research Vessel Captains & Polar Expedition Planners	Voyage-specific, explainable navigation risk scoring and NetworkX-based safe-path planning around forecasted iceberg hazard zones	—
Maritime & Polar Researchers	Iceberg drift tracking data with direct scientific value for glaciology and ice-shelf dynamics research	Reproducible SAR + physics-modeling architecture extensible to other ocean-object tracking problems
Coast Guards & Maritime Surveillance Authorities (ICG / USCG / EMSA)	Extendable to polar/sub-polar surveillance zones as ice-covered shipping routes expand	Detect offshore pollution events and generate forensic-grade evidence dossiers
Environmental Protection Agencies / Pollution Control Boards	—	Hold polluters accountable for illegal bilge dumps
Port & Harbor Authorities	—	Monitor near-shore anchorage zones for compliance

✨ Key Features
Shared / Core Platform
🛰️ Deep Learning SAR Segmentation — ResNet-34 U-Net (transfer-learned, PyTorch) segments radar anomalies from Sentinel-1 imagery.
🧠 Hazard Type Classifier — a dedicated classifier distinguishes detected anomalies into Oil Slick / Iceberg / False Alarm, trained on real labeled SAR data (Statoil/C-CORE iceberg dataset).
🔄 Unified Drift Engine — OpenDrift-based Lagrangian physics, run backward (oil origin) or forward (iceberg forecast) depending on hazard type.
💻 Interactive Full-Stack Console — Vite + Leaflet-style geospatial dashboard, FastAPI backend, PostgreSQL + PostGIS storage.
Oil Spill Path
🚢 Transparent Multi-Factor Vessel Attribution — 5 independently-weighted, explainable signals (spatial, temporal, trajectory, drift-consistency, speed/course) rank candidate vessels — no black-box score.
🔐 Court-Ready Forensic PDF Export — client-side SHA-256 hash + QR-verifiable case seal.
Iceberg Path
📍 24–72 Hour Drift Forecasting — forward Lagrangian simulation with 68%/95% uncertainty ellipses per timestep.
⚠️ Route-Risk Scoring — evaluates a ship's planned route against forecasted iceberg hazard zones, flags segments Low/Medium/High risk with a plain-language explanation.
🧭 Safe-Path Planning (Module 4) — NetworkX graph-based pathfinding recommends the lowest-risk route around multiple forecasted iceberg hazards, with distance/detour and clearance metrics.
🏗️ Architecture & Design Principles
Additive, non-destructive extension — The iceberg capability was built entirely as new files and new branches alongside the original oil-spill modules. No existing oil-spill model weights, training pipelines, or attribution logic were modified; all original test suites pass unchanged.
Contract-driven branching — A single hazard_type field in the detection contract ("oil" / "iceberg" / "false_alarm") determines which downstream path runs, keeping both hazard pipelines independently testable.
Real data, not synthetic — Oil-spill model trained on a published Sentinel-1 SAR benchmark and validated against a real post-Hurricane-Ida Gulf of Mexico scene; iceberg classifier trained on the real Statoil/C-CORE SAR-labeled dataset.
Explainability over black-box confidence — Both the vessel-attribution score and the route-risk score are transparent, auditable, and signal-by-signal explainable — critical for any real investigative, legal, or navigational safety use.

🚀 Setup & Run Guide
📋 Prerequisites
Git, Python 3.10+, Node.js 18+ & npm
PostgreSQL (optional — required only for persistent incident/route storage)
Step 1 — Clone the Repository
bash
git clone https://github.com/senvidit4-alt/Ocean-Trace-v2.0.git
cd Ocean-Trace-v2.0
Step 2 — Backend Setup

Windows (PowerShell):

powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py

Linux / macOS:

bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py

✅ Backend live at: http://localhost:8000 📖 Interactive Swagger Docs: http://localhost:8000/docs 🩺 Health Check: http://localhost:8000/api/health

Step 3 — Frontend Setup
bash
cd ocean-trace-frontend
npm install
npm run dev

🖥️ Web Console live at: http://localhost:5173

Step 4 — (Optional) Database Setup
bash
cp .env.example .env
# Fill in your PostgreSQL credentials in .env
python report_db.py
🧪 Automated Verification & Testing
bash
python test_backend_api.py

Expected output:

============================================================
ALL API ENDPOINT INTEGRATION TESTS PASSED SUCCESSFULLY!
============================================================

Run the full pipeline end-to-end via CLI (oil-spill path by default):

bash
python run_pipeline.py --input-synthetic --output-dir outputs

Run the iceberg path (manual override until the hazard classifier is fully wired):

bash
python run_pipeline.py --hazard-type iceberg --output-dir outputs
📁 Repository Structure
Ocean-Trace-v2.0/
├── backend/
│   └── main.py                      # FastAPI endpoints — oil-spill + iceberg routes
├── ocean-trace-frontend/            # Interactive Web UI Console
├── modules/
│   ├── 01_detection/
│   │   └── src/
│   │       ├── model.py             # Oil-spill ResNet-34 U-Net
│   │       ├── data_loader.py
│   │       ├── train.py
│   │       └── iceberg_classifier.py  # Hazard-type classifier
│   ├── 02_drift/
│   │   └── src/
│   │       ├── source_reconstruction.py  # Oil backward hindcast
│   │       └── iceberg_forecast.py       # Iceberg forward forecast
│   ├── 03_attribution/
│   │   └── src/
│   │       ├── evidence_fusion.py   # Oil vessel attribution
│   │       └── route_risk.py        # Iceberg route-risk scoring
│   └── 04_navigation/                # Iceberg navigation module
│       └── src/
│           └── safe_path.py         # NetworkX safe-path planning
├── data/
│   ├── AIS_*.csv                    # Real Gulf of Mexico AIS telemetry
│   └── iceberg_dataset/
│       └── train.json               # Statoil/C-CORE iceberg SAR training data
├── contracts/                       # JSON schemas for inter-module handoffs
├── docs/                            # Architecture specifications
├── unet_spill_best.pth              # Pretrained oil-spill U-Net weights
├── iceberg_classifier_best.pth      # Trained iceberg hazard classifier
├── run_pipeline.py                  # Full CLI pipeline runner — branches by hazard_type
├── test_backend_api.py              # Integration test suite
├── requirements.txt
├── .env.example                     # Environment variable template (no real secrets)
└── main.py

🔒 Note: .env files, API keys, and database credentials are never committed to this repository — only .env.example templates are tracked. See Security below.

⚙️ REST API Endpoints
Method	Endpoint	Description
GET	/health	Service health status
POST	/detect-spill	Module 1 — unified SAR detection, returns hazard_type
POST	/trace-origin	Module 2 (oil) — backward hindcast
POST	/attribute-vessel	Module 3 (oil) — AIS vessel attribution
POST	/score-route-risk	Module 3 (iceberg) — route-risk scoring (new)
POST	/find-safe-path	Module 4 (iceberg) — safe-path planning (new)
POST	/run-full-pipeline	Full pipeline — auto-branches by detected hazard_type

Full interactive API reference at /docs once the backend is running.

🧰 Tech Stack
Layer	Technology
Detection	PyTorch, segmentation_models_pytorch (ResNet-34 U-Net), hazard classifier CNN, rasterio, OpenCV
Drift & Forecasting	OpenDrift (backward hindcast + forward forecast), NetCDF, CMEMS / ECMWF
Oil Attribution	pandas, GeoPandas, Shapely, NOAA MarineCadastre AIS data
Iceberg Route Risk & Navigation	GeoPandas, Shapely, NetworkX (graph-based safe-path routing)
Backend	FastAPI, Uvicorn, Pydantic
Database	PostgreSQL, psycopg2, PostGIS, JSONB
Frontend	Vite, HTML/CSS/JavaScript, Leaflet-style map rendering
Forensic Export	html2pdf.js, Web Crypto API (SHA-256), qrcode.js
Deployment	Render (backend), Vercel (frontend)
🛠️ Troubleshooting
PowerShell execution policy error (Windows) — Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass before activating the venv.
Port conflicts — run backend on another port via uvicorn main:app --port 8080 and update VITE_API_URL accordingly.
Model weights — both unet_spill_best.pth and iceberg_classifier_best.pth are included in the repo root; the backend auto-loads both at startup.
Database errors — the app runs fully without PostgreSQL, just without persistent incident/route storage.
🔒 Security
No secrets in source control — database credentials, API keys, and connection strings are read from environment variables (.env, excluded via .gitignore); only .env.example templates with placeholder values are committed.
Parameterized database queries — all PostgreSQL inserts/queries use parameterized statements (psycopg2 with %s placeholders), not string-formatted SQL, to prevent injection.
Input validation at the API boundary — FastAPI endpoints use Pydantic models to validate and type-check all incoming request data before it reaches any detection, drift, attribution, or routing logic.
File upload validation — uploaded SAR imagery is checked for valid format/content before processing, rejecting malformed or oversized files.
CORS scoped for deployment — the backend's CORS policy is restricted to the deployed frontend origin in production rather than left fully open.
Tamper-evident reporting — forensic PDF exports are sealed with a client-side SHA-256 hash and QR-verifiable case ID, so any report can be checked for post-generation tampering.
No credentials or personal data in logs — application logs record operational events (detections, pipeline runs) without exposing database credentials or sensitive request payloads.
🔮 Future Scope
Live global AIS traffic integration for continuous maritime monitoring.
Extending the hazard classifier to additional SAR look-alike categories (algal blooms, biogenic films) for even lower false-positive rates.
Deployment against live Antarctic and Indian Ocean satellite coverage as a real-time regional monitoring roadmap.
👥 Team

Team Ocean Trace — Smart India Hackathon 2026 Problem Statement: SIH26059 (Ministry of Earth Sciences / NCPOR) Repository: github.com/senvidit4-alt/Ocean-Trace-v2.0

📄 License

This project is licensed under the MIT License.
