# Project Context: Adaptive Geospatial System for Autonomous Mining Monitoring

> **Team Rogue | AURORA 2.0 — First-Place Solution**  
> **Repository:** `Sameer513-code/Team-Rogue-Aurora-Mining-Monitor`  
> **Domain:** Remote Sensing, Earth Observation, Geospatial Decision Support Systems (DSS), Environmental Compliance  
> **Key Technologies:** Python, Google Earth Engine (GEE), Sentinel-2 (Optical), Sentinel-1 (SAR), FastAPI, Leaflet.js, Chart.js, NumPy

---

## 1. Executive Summary & Purpose

### What is this project about?
The **Adaptive Geospatial System for Autonomous Mining Monitoring** is an end-to-end Geospatial Decision Support System (DSS) developed for automated regulatory monitoring of open-cast / open-pit mining activities and adherence to geo-fenced **No-Go Zones** (protected boundaries, ecological reserves, river corridors, and village buffers).

### The Problem
1. **Infrequent & High-Risk Audits:** Manual ground inspections of mines are hazardous, sporadic, expensive, and subject to jurisdiction-wide regulatory blind spots or corruption.
2. **Brittleness of Traditional Remote Sensing:** Static spectral index thresholds (e.g., fixed $\text{NDVI} < 0.2$) break down when applied across differing mineral geologies (e.g., Coal vs. Bauxite vs. Limestone), differing soil types, seasonal vegetation phenology, and changing sun angles.
3. **Cloud Obstruction:** Optical satellites are blind during heavy cloud cover and monsoon seasons—precisely when unauthorized encroachment often accelerates.
4. **Black-box AI Limitations:** Machine learning (e.g., deep CNNs) requires extensive labeled training data across varied geographies, lacks physical transparency, suffers from hallucinations/overfitting, and fails to produce legally defensible, explainable audit trails.

### The Solution
Team Rogue built a **fully unsupervised, mine-agnostic, physics-informed satellite surveillance pipeline**:
- Fuses **Sentinel-2 multispectral optical** and **Sentinel-1 SAR radar** data to ensure all-weather, cloud-penetrating coverage.
- Uses **Adaptive Percentile-Based Change Vector Analysis (CVA)** ($85^{\text{th}}$ percentile for optical, $80^{\text{th}}$ percentile for SAR) to dynamically calibrate change thresholds per temporal interval from local distribution statistics rather than fixed numbers.
- Enforces a **Temporal Stabilization & Irreversibility Rule** ("holes don't disappear") requiring multi-month persistence, eliminating transient noise (agricultural tilling, cloud shadows, phenological shifts).
- Provides an **interactive decision-support dashboard** with time-travel scrubbing, quantitative KPIs ($km^2$, growth rates, linear predictions), true-color quantified segmentation overlays, and an immutable chronological violation audit log.

---

## 2. Technical Deep Dive: The Science & Algorithms

```
[Sentinel-2 L2A (Optical)]        [Sentinel-1 GRD (SAR)]
       │                                   │
  QA60 Cloud Mask (Cloud < 10%)       IW Descending (VV, VH, VV-VH)
       │                                   │
Monthly Median Composite (NDVI, NBR)  Monthly Median Composite (VV, VH, Ratio)
       │                                   │
       └──────────────┬────────────────────┘
                      ▼
     Change Vector Analysis (CVA) Differencing
         ΔIt = It - It-1 (Optical & SAR)
                      │
                      ▼
        Adaptive Percentile Thresholding
       (85th %ile Optical, 80th %ile SAR)
                      │
                      ▼
        Multi-Sensor Fusion (Logical OR)
            Mcandidate = Mopt ∨ MSAR
                      │
                      ▼
        Temporal Consistency & Stabilization
           Mstable = Mcandidate(t) ∧ Mcandidate(t-1)
                      │
                      ▼
         Cumulative Masking (Persistence)
              Mcum(t) = Mcum(t-1) ∨ Mstable(t)
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
[Analytical Area Metric]    [Visual Map Generation]
  Raw Mcum · pixelArea()      Mcum ∧ (NDVI ≤ 0.3)
  Area (km²), Growth, Regr.   Masked True-Color RGB PNG
```

### 2.1 Satellite Data Sources & Pre-Processing
1. **Optical — Copernicus Sentinel-2 Level-2A (Harmonized):**
   - **Bands:** B4 (Red, 665nm), B8 (NIR, 842nm), B11 (SWIR-1, 1610nm), B2/B3/B4 (True color RGB).
   - **Quality Masking:** Pixels with opaque clouds (bit 10) or cirrus (bit 11) in the `QA60` quality band are masked out. Scenes with $>10\%$ total cloud cover are discarded.
   - **Scale:** Reflectance values divided by 10,000 ($0.0 \text{ to } 1.0$).
   - **Indices Computed:**
     - $\text{NDVI} = \frac{B8 - B4}{B8 + B4}$ (Sensitive to photosynthetic vegetation loss).
     - $\text{NBR} = \frac{B8 - B11}{B8 + B11}$ (Normalized Burn Ratio; sensitive to exposed bare earth, rock, and moisture reduction).
2. **SAR — Copernicus Sentinel-1 Ground Range Detected (GRD):**
   - **Configuration:** C-band Synthetic Aperture Radar, Interferometric Wide (IW) swath mode, Descending orbit pass.
   - **Bands:** Dual polarization: VV (sensitive to surface roughness and geometric modifications) and VH (sensitive to volume scattering and vegetation degradation).
   - **Polarization Ratio:** $\text{RATIO} = VV - VH$ (difference in dB, accentuating ground disturbance).
   - **Weather Resilience:** Microwave pulses penetrate clouds, fog, haze, and rain.

### 2.2 Adaptive Change Vector Analysis (CVA)
Rather than evaluating absolute index values, excavation is detected via **magnitude of temporal deviation** between consecutive observation composites ($t-1$ and $t$):

$$\Delta I_t = I_t - I_{t-1}$$

1. **Optical CVA Magnitude ($M_{\text{optical}}$):**
   Euclidean distance in spectral index space:
   $$M_{\text{optical}} = \sqrt{(\Delta \text{NDVI})^2 + (\Delta \text{NBR})^2}$$
   *(The extended research formulation also incorporates $\Delta \text{SWIR}$).*
2. **SAR CVA Magnitude ($M_{\text{SAR}}$):**
   Manhattan L1-norm across roughness and ratio metrics:
   $$M_{\text{SAR}} = |\Delta VV| + |\Delta VH| + |\Delta \text{RATIO}|$$

### 2.3 Adaptive Percentile Thresholding (Self-Calibrating)
Fixed thresholds fail across sites with different mineral signatures (dark coal vs reddish bauxite vs grey limestone). Team Rogue's system computes dynamic thresholds directly from the statistical distribution of changes across the mine geometry for each interval:

$$T_{\text{optical}} = P_{85}(M_{\text{optical}})$$
$$T_{\text{SAR}} = P_{80}(M_{\text{SAR}})$$

- Pixels exceeding these thresholds are marked as **candidate disturbances**:
  $$M_{\text{candidate}} = (M_{\text{optical}} > T_{\text{optical}}) \lor (M_{\text{SAR}} > T_{\text{SAR}})$$
- **Why relaxed OR fusion?** If cloud cover renders optical data degraded or absent, SAR guarantees continuity. If SAR suffers radar shadow or geometric distortion along high pit walls, optical provides multispectral discrimination.

### 2.4 Temporal Stabilization & Physical Irreversibility
Open-cast mining involves irreversible topographic removal. Once pit excavation occurs, bedrock or spoil does not spontaneously turn back into pristine forest within a couple of months:
1. **Consecutive Detection Requirement:**
   $$M_{\text{stable}}^t = M_{\text{candidate}}^t \land M_{\text{candidate}}^{t-1}$$
   *(A candidate pixel is confirmed only if disturbance was also present in the prior step).*
2. **Cumulative Persistence Mask:**
   $$M_{\text{cum}}^t = M_{\text{cum}}^{t-1} \lor M_{\text{stable}}^t$$
   *(Monotonically accumulating; prevents temporal flickering and discards non-persistent events such as agricultural harvesting or cloud shadow).*

### 2.5 Separation of Analytical Metrics vs. Visualization Maps
- **Analytical Calculations (Strict Raw Mask):** Area calculations strictly utilize $M_{\text{cum}}$ without secondary suppression to preserve true physical disturbance measurements:
  $$\text{Area (km}^2) = \frac{\sum (M_{\text{cum}} \times \text{pixelArea})}{10^6}$$
- **Quantified Map Generation (Vegetation Suppression):**
  $$M_{\text{viz}} = M_{\text{cum}} \land (\text{NDVI} \le 0.3)$$
  Masks the Sentinel-2 true-color RGB image so that non-excavated or re-vegetated areas drop to black, while active pits, benches, haul roads, and dumps are highlighted in photographic true color for human inspectors.

### 2.6 Compliance Classification & Predictive Trend Analysis
- **Alert Tiers:**
  - $0\% \text{ mined}$ in No-Go Zone $\to$ `none` (Compliant $\🛡️$)
  - $< 1\% \text{ mined}$ $\to$ `soft` alert
  - $\ge 1\% \text{ mined}$ (or $> 0.002\text{ km}^2$) $\to$ `hard` alert (`VIOLATION DETECTED` ⚠️)
- **First Violation Timestamping:** Finds the earliest observation date where illegal excavated area $> 0$.
- **Predictive Early Warning:** Uses linear regression (`np.polyfit(x, y, deg=1)`) over historical cumulative data points to extrapolate the expected excavated area for the upcoming month:
  $$y_{\text{next}} = m \cdot (N) + c$$

---

## 3. System Architecture & Component Breakdown

```
┌────────────────────────────────────────────────────────┐
│                   CLIENT BROWSER                       │
│  [Upload GeoJSON] ──> [Leaflet Map + Chart.js + Logs]  │
└───────────▲───────────────────────────────▲────────────┘
            │ HTTP POST /run (GeoJSON)      │ HTTP GET /results
            │ HTTP GET /progress            │ HTTP GET /static/*.png
┌───────────▼───────────────────────────────▼────────────┐
│               FASTAPI BACKEND (app.py)                 │
│  - Endpoint Router (/run, /progress, /results, /static)│
│  - Asynchronous BackgroundTasks Worker                 │
└───────────────────────────┬────────────────────────────┘
                            │ Calls run_pipeline()
┌───────────────────────────▼────────────────────────────┐
│            GEOSPATIAL PIPELINE (pipeline.py)           │
│  - Service Account Authentication (key.json)           │
│  - Batched GEE Reductions & Temporal Filtering         │
│  - CVA, Thresholding, Stabilization & Metrics          │
│  - Exports Thumbnail Maps to disk (outputs/*/*.png)    │
│  - Serializes Results to output.json                   │
└───────────────────────────┬────────────────────────────┘
                            │ REST API Calls / Graph Evaluation
┌───────────────────────────▼────────────────────────────┐
│             GOOGLE EARTH ENGINE (GEE) CLOUD            │
│  - COPERNICUS/S2_SR_HARMONIZED                         │
│  - COPERNICUS/S1_GRD                                   │
└────────────────────────────────────────────────────────┘
```

### File-by-File Breakdown

| File / Directory | Purpose & Key Contents |
| :--- | :--- |
| `pipeline.py` | **Core Geospatial Engine:** Implements `init_ee()`, S2/S1 collection filters, CVA math, batched reductions, `run_detection()`, metric computers (`compute_monthly_growth`, `predict_next_month`, `classify_alert`), and the top-level `run_pipeline()`. |
| `app.py` | **FastAPI Server:** Exposes endpoints `POST /run`, `GET /progress`, `GET /results`, mounts static directory `/static` pointing to `./outputs`, and manages background pipeline execution. |
| `main_pipeline.ipynb` | **Jupyter Experimentation Notebook:** Prototype notebook used during development/Colab. Contains sanity-check assertions, matplotlib sanity plots, array shape broadcasting tests, and thumbnail generation. |
| `frontend/index.html` | **Single-Page Application (SPA):** Contains the 3 screens (`uploadScreen`, `loaderScreen`, `resultsScreen`), KPI cards, Leaflet map container, Chart canvas, temporal slider, and quantified modal overlay. |
| `frontend/script1.js` | **Client Logic:** Handles GeoJSON upload parsing via `FileReader`, triggers `/run`, polls `/progress` every 1000ms, calls `/results`, updates Leaflet polygons, draws Chart.js graphs, updates reverse-chronological violation logs, and toggles quantified image overlays. |
| `frontend/style.css` | **Modern Dark Cyberpunk / Industrial Theme:** Custom dark UI (`#0a0d16`), animated grid backdrop, glassmorphism cards, glowing status borders (`#10b981` success, `#ef4444` danger), smooth CSS transitions. |
| `sample inputs/` | `legal_boundary.geojson` and `no_go_zone.geojson` representing a real open-cast limestone/coal mining site (near Rewa/Maihar, MP, India at coordinates ~24.377°N, 81.407°E). |
| `example outputs/` | Pre-computed PNG thumbnails for `mine/` and `no_go_zone_0/` spanning from `2020-03-01.png` to `2024-11-01.png`. |
| `End Term Report.pdf` | Academic research report detailing the mathematical formulation, CVA equations, experimental results, and regulatory utility. |
| `Presentation.pdf` | 12-slide pitch deck explaining the operational challenges, sensor choices, CVA algorithms, temporal stabilization, and system robustness. |

---

## 4. Engineering Optimizations: Eliminating GEE Bottlenecks

In initial implementations (and typical GEE beginner scripts), calling `.getInfo()` inside loops causes separate network round-trips for every month, band, and polygon. For a 5-year bi-monthly analysis across multiple zones, this resulted in hundreds of round-trips taking **15 to 30+ minutes** or hitting GEE quota limits.

Team Rogue implemented **three key server-side batching optimizations** in `pipeline.py`:

1. **Batch Validation Check:**
   Instead of looping through all candidate months and executing `.getInfo()` to verify data availability one-by-one:
   ```python
   # Single network round-trip evaluates image collection sizes across all candidate months
   band_counts = ee.List([img.bandNames().size() for img in check_imgs]).getInfo()
   ```
2. **Global Threshold Pre-Computation on Mine Boundary:**
   Percentile reductions ($P_{85}$ for optical, $P_{80}$ for SAR) are compiled into a server-side list and resolved in a single network query:
   ```python
   threshold_results = ee.List(threshold_ops).getInfo()
   ```
   These thresholds are cached and reused across the mine and all No-Go zones without re-computing per zone.
3. **Batched Area Reduction Promises:**
   Area integrals (`reduceRegion(ee.Reducer.sum())`) inside the time-series loop are stored as server-side `ee.Number` promises without calling `.getInfo()` inside the loop:
   ```python
   resolved_areas = ee.List(area_promises).getInfo()
   ```
   *Result:* Total execution time dropped from **~20 minutes to 2–4 minutes** for a 5-year multi-sensor analysis.

---

## 5. API Specification & Data Contracts

### 5.1 Endpoints (`app.py`)

- **`POST /run`**
  - **Body:**
    ```json
    {
      "mine_geojson": { "type": "FeatureCollection", "features": [...] },
      "no_go_geojson_list": [
        { "type": "FeatureCollection", "features": [...] }
      ]
    }
    ```
  - **Response:** `{"status": "started"}`
  - **Behavior:** Clears `ANALYSIS_RESULT`, resets `PIPELINE_PROGRESS`, and triggers `background_wrapper` asynchronously.

- **`GET /progress`**
  - **Response:**
    ```json
    {
      "status": "idle" | "running" | "done" | "error",
      "progress": 0 to 100,
      "error": null | "error message"
    }
    ```

- **`GET /results`**
  - **Response:** Returns the full `output.json` payload (see schema below).

- **`GET /static/{out_id}/{date_key}.png`**
  - Serves generated quantified true-color PNGs from local `./outputs/`.

### 5.2 Schema of `output.json`
```json
{
  "metadata": {
    "analysis_start": "2020-03-01",
    "analysis_end": "2024-11-01",
    "valid_months": [[2020, 1], [2020, 3], [2020, 5], "..."]
  },
  "mine": {
    "timeseries": [
      { "date": "2020-03-01", "area_km2": 0.0306 },
      { "date": "2020-05-01", "area_km2": 0.0342 }
    ],
    "current_area_km2": 0.0770,
    "percentage_mined": 47.26,
    "monthly_growth": [
      { "date": "2020-05-01", "growth_km2": 0.0036 }
    ],
    "current_month_growth": 0.0012,
    "predicted_next_month_area": 0.0782,
    "quantified_maps": {
      "2020-03-01": "/static/mine/2020-03-01.png",
      "2024-11-01": "/static/mine/2024-11-01.png"
    }
  },
  "no_go_zones": {
    "no_go_zone_0": {
      "timeseries": [
        { "date": "2020-03-01", "area_km2": 0.0 },
        { "date": "2020-05-01", "area_km2": 0.0004 }
      ],
      "current_area_km2": 0.0026,
      "percentage_mined": 3.25,
      "alerts": [
        {
          "date": "2020-05-01",
          "area_km2": 0.0004,
          "growth_km2": 0.0004,
          "alert": "hard"
        }
      ],
      "first_violation": "2020-05-01",
      "monthly_growth": [...],
      "predicted_next_area": 0.0028,
      "predicted_next_alert": "hard",
      "analysis_start": "2020-03-01",
      "analysis_end": "2024-11-01"
    }
  }
}
```

---

## 6. Frontend Dashboard Interaction Flow

1. **Upload Screen:**
   - User uploads `legal_boundary.geojson` and `no_go_zone.geojson`.
   - File inputs validate JSON structure. The "RUN ANALYSIS" button lights up once both files are loaded.
2. **Loader Screen:**
   - Displays real-time progress bar (0% to 100%) synchronized via 1-second polling on `/progress`.
3. **Results Screen:**
   - **Context Tab Switcher:** Toggles between `Legal Zone` and `No-Go Zone`. All KPI cards, chart curves, and alerts re-contextualize dynamically based on the selected tab.
   - **KPI Cards:**
     - In *Legal* Mode: Current Legal Area, Monthly Growth %, Predicted Next Month Area.
     - In *No-Go* Mode: Illegal Mining Area, Illegal Growth %, Predicted Next Month Encroachment.
   - **Interactive Leaflet Map:** Renders satellite base layer (Esri World Imagery) with green outline for legal boundaries and dashed red outline for No-Go zones. Auto-fits bounds to the uploaded geometry.
   - **Time-Travel Slider:** Allows scrubbing through all historical bi-monthly dates (2020 to 2024). Scrubbing updates:
     - The top status bar (`OK` vs `VIOLATION DETECTED`).
     - Real-time KPI values at that exact point in time.
     - Chart history up to that date.
     - Reverse-chronological audit log.
     - Quantified segmentation map image.
   - **Chart.js Timeline:** Displays cumulative area growth curve. Supports toggling datasets dynamically between legal and illegal zones.
   - **Quantified Map Viewer:** Toggled via "View Quantified Maps" button. Replaces vector map canvas with true-color satellite imagery segmented to show actual excavated rock, pits, and tailings.
   - **Violation Log (Immutable Audit Trail):** Lists chronological entries with log IDs (e.g. `LOG_ID: 0005`), violation timestamps, recorded illegal area, and threshold excess warnings.

---

## 7. Key Assumptions, Design Decisions & Constraints

| Factor | Design Decision | Rationale |
| :--- | :--- | :--- |
| **Algorithm Choice** | Unsupervised CVA + Adaptive Percentile vs Deep Learning (CNN/U-Net) | Regulatory compliance demands **explainability and auditability**. Supervised ML requires scarce, noisy ground truth, overfits to specific mineral colors, and produces uninterpretable probabilities rather than physical surface changes. |
| **Thresholding** | Dynamic Percentiles ($85^{\text{th}} / 80^{\text{th}}$) vs Fixed Values | Mineral surfaces vary drastically in spectral response. Statistical percentiles self-calibrate to local scene contrast and seasonality. |
| **Persistence Filter** | 2 consecutive observation periods + cumulative logical OR | Completely removes agricultural plowing, seasonal leaf fall/greening, ephemeral puddles, and cloud shadows. |
| **Temporal Cadence** | Bi-monthly (Jan, Mar, May, Jul, Sep, Nov) | Balances GEE compute budget with Sentinel revisit schedules, avoiding cloudiest monsoon intervals while capturing operational mining progression. |
| **Spatial Resolution**| 10 meters | Matches the native resolution of Sentinel-2 VNIR bands (B2, B3, B4, B8) and Sentinel-1 IW GRD pixels. |
| **System Decoupling** | GEE backend generates `output.json`; frontend consumes it | Allows field officers in low-connectivity remote mining outposts to inspect and interact with the full 5-year audit history without a fast internet connection. |

---

## 8. Setup, Configuration & How to Run

### Prerequisites
- Python 3.9+ (Tested on Python 3.11/3.12)
- An active **Google Earth Engine (GEE)** enabled Google Cloud project.

### 1. GEE Authentication
1. Service Account (Recommended):
   - Generate a service account key JSON from Google Cloud Console with Earth Engine API access enabled.
   - Place the key file in the repository root as `key.json`.
   - `pipeline.py` automatically loads credentials from `key.json`.
2. Interactive Auth (Alternative):
   - Run `earthengine authenticate` in your terminal and initialize with `ee.Initialize(project='your-gcp-project-id')`.

### 2. Python Environment & Installation
```bash
# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running Backend & Frontend
The application runs as two decoupled servers:

**Terminal 1 (Backend FastAPI API):**
```bash
python app.py
# Runs on http://127.0.0.1:8000
```

**Terminal 2 (Frontend Client):**
```bash
cd frontend
python -m http.server 3000
# Accessible at http://localhost:3000
```

### 4. Running the Standalone Notebook
To inspect sanity tests, data arrays, or run standalone experiments:
```bash
jupyter notebook main_pipeline.ipynb
```

---

## 9. Quick Reference for Future AI Agents & Developers

- **To adjust detection sensitivity:**
  - Optical percentile threshold: `pipeline.py` line 396 (`ee.Reducer.percentile([85])`). Lowering to 80 increases sensitivity; raising to 90 restricts to more extreme disturbances.
  - SAR percentile threshold: `pipeline.py` line 399 (`ee.Reducer.percentile([80])`).
- **To change date range or cadence:**
  - `pipeline.py` lines 331–332: `years = range(2020, 2025)` and `months = [1, 3, 5, 7, 9, 11]`.
- **To alter violation threshold in the frontend:**
  - `frontend/script1.js` lines 207 & 284: `VIOLATION_LIMIT = 0.002` (in $km^2$).
- **To change visualization brightness:**
  - `pipeline.py` line 80: `max=0.3` in `export_png()`. Increase to 0.4–0.5 if satellite images appear too dark.
- **Key data artifact:**
  - The pipeline outputs `output.json` at the root and image sets into `outputs/{out_id}/{date}.png`.
