# Battery RUL AI Inference System

Deep learning-based full-stack AI inference system for battery remaining useful life (RUL) prediction, live reinference, and degradation monitoring.

**Goal**: Build an end-to-end AI inference system that predicts battery RUL from early-cycle observations and visualizes degradation signals through a web dashboard.

**Live Demo**: https://onekindalpha-battery-rul-dashboard-bmaml-svgd.hf.space  
**Demo Video**: https://github.com/user-attachments/assets/1e05d64d-b9e3-47ac-abc4-06048ae3b7a7

The demo shows cycle-level RUL inference, degradation monitoring, battery comparison, explainability with pre-EOL anomaly evidence, SHAP-based feature importance, and live reinference.

---

## TL;DR

- Built an end-to-end deep learning inference system for NASA battery RUL prediction and degradation monitoring.
- Designed domain-informed time-series features and few-shot support/query tasks from battery cycle data.
- Implemented BMAML-SVGD-style uncertainty-aware meta-learning with a CEEMDAN-Transformer-DNN backbone.
- Built and deployed a FastAPI + React dashboard with live reinference, uncertainty visualization, degradation monitoring, SHAP-based explainability, precomputed baseline restoration, and frontend CSV download for the currently displayed prediction result.
- Used DuckDB-backed local feature data access with CSV/Parquet support and local JSON prediction payloads to support dashboard data loading and precomputed inference results.

---

## Project Overview

This project is an end-to-end applied deep learning system that connects battery domain knowledge, sequence modeling, inference APIs, and a deployable web dashboard.

- **Problem**: Predict battery remaining useful life from early-cycle observations and monitor degradation behavior.
- **Data Pipeline**: NASA battery cycle data → domain-informed feature engineering → few-shot support/query task construction.
- **Modeling**: BMAML-SVGD-style few-shot Bayesian meta-learning with a CEEMDAN-Transformer-DNN backbone.
- **Inference System**: PyTorch model inference with precomputed cache and on-demand live reinference.
- **Application**: FastAPI backend + React/Vite dashboard deployed with Docker on Hugging Face Spaces.

---

## My Role

I implemented the core pipeline from data preprocessing to model inference, dashboard development, and deployment.

- **Data / Feature Engineering**: NASA battery preprocessing, 40 time-series features, support/query task construction, and robust scaling.
- **Deep Learning**: BMAML-SVGD-style meta-learning, CEEMDAN-Transformer-DNN backbone, and uncertainty estimation.
- **Backend**: FastAPI inference APIs, DuckDB-backed local feature data access with CSV/Parquet support, degradation monitoring endpoints, and precomputed/live inference result handling.
- **Frontend**: React dashboard, RUL curve visualization, uncertainty band rendering, degradation tabs, and live reinference state management.
- **Deployment**: Dockerized full-stack app deployment on Hugging Face Spaces with Git LFS checkpoint handling.

---

## Key Features

### 1. RUL Prediction Dashboard

- Visualizes predicted RUL trajectory from the support period to the query period.
- Displays prediction curves, ground-truth references, and model uncertainty bands.
- Calculates RMSE, MAE, and confidence-related metrics for selected batteries and observation ratios.

### 2. Live Reinference

- Provides an `Initialize & Reinference` flow for on-demand model inference.
- Runs CPU-based reinference in the deployed Hugging Face Spaces environment.
- Updates prediction curves, uncertainty values, confidence metrics, and dashboard state after reinference.
- Allows users to restore the precomputed baseline result after live reinference.
- Allows users to download the currently displayed prediction curve as a CSV file.

### 3. Degradation Monitoring

- Monitors degradation-related signals such as capacity, DCR, impedance, temperature, current stress, and LLI proxy signals.
- Uses cycle-level evidence to identify abnormal degradation behavior.
- Provides a degradation-focused view beyond a single RUL prediction number.

### 4. Explainability & Uncertainty

- Estimates prediction uncertainty using multiple SVGD-style prediction particles.
- Visualizes global feature importance through SHAP-based analysis.
- Connects model output with degradation-related feature behavior.

### 5. Deployment-Oriented Inference Flow

- Uses precomputed prediction cache for fast initial loading.
- Supports live reinference when deeper analysis is needed.
- Handles local and Docker/Hugging Face runtime path differences through deployment-aware configuration.

---

## System Architecture

```text
Raw NASA Battery Data
    ↓
[Data Engineering]
    - CSV / cycle-level data processing
    - Domain-informed feature engineering
    - 40 time-series features
    - Robust scaling and support/query task construction
    ↓
[Model Training]
    - BMAML-SVGD-style few-shot meta-learning
    - CEEMDAN-Transformer-DNN backbone
    - Ray Tune + ASHA hyperparameter tuning
    - Checkpoint: nasa_bmaml_best_re.pt
    ↓
[Precomputed Cache]
    - Cached predictions for multiple r_ratio settings
    - JSON payloads for fast dashboard loading
    - Fallback path when live inference is unavailable
    ↓
[Live Inference Engine]
    - POST /api/battery/{id}/reinfer?r_ratio=X
    - CPU-based on-demand inference
    - Output: prediction, uncertainty, confidence, and metrics
    ↓
[FastAPI Backend]
    - Inference APIs
    - DuckDB-backed local feature data access with CSV/Parquet support
    - Local JSON prediction payload loading
    - Degradation monitoring
    - Precomputed/live inference result handling
    ↓
[React Frontend]
    - Overview tab
    - Degradation tab
    - Compare tab
    - Explainability view
    - Live reinference controls
    ↓
[Docker / Hugging Face Spaces Deployment]
    - Full-stack app serving
    - Git LFS checkpoint handling
    - Deployment-aware file path configuration
```

---

## Deep Learning Approach

### Few-Shot RUL Prediction

In real battery operation, long-term degradation data may not be available at the beginning of a battery's life. This project frames early-cycle RUL estimation as a few-shot prediction problem.

```text
Support Set: early observed cycles
Query Set: future cycles to predict
Task: adapt to each battery using limited early-cycle information
```

### BMAML-SVGD-Style Meta-Learning

The model is inspired by Bayesian MAML and SVGD-based uncertainty estimation.

```text
Input: sequence features + summary features
    ↓
CEEMDAN-based signal decomposition features
    ↓
Transformer encoder for temporal dependency modeling
    ↓
DNN prediction head for RUL estimation
    ↓
SVGD-style particles for uncertainty-aware predictions
```

### Why This Approach

- **Few-shot adaptation** helps model new batteries from limited early-cycle observations.
- **Sequence modeling** captures degradation patterns across time.
- **Uncertainty estimation** provides more information than a single point prediction.
- **Domain-informed features** connect ML predictions with physical degradation behavior.

---

## Data Pipeline

### Feature Engineering

Raw battery measurements are transformed into degradation-related time-series features.

```text
Raw Battery Data
    - Voltage
    - Current
    - Temperature
    - Capacity
    - Impedance
    ↓
Time-Series Features
    - Capacity degradation velocity
    - DCR / impedance growth rate
    - Temperature stress indicators
    - Current / load stress metrics
    - CEEMDAN-based IMF decomposition features
    - LLI / LAM proxy signals
    ↓
Few-Shot Task Construction
    - Support set
    - Query set
    - Task-level sampling
```

### Main Data Components

- **Feature count**: 40 time-series features
- **Scaling**: custom robust 3D scaling for sequence data
- **Task setup**: support/query construction for meta-learning
- **Data access**: DuckDB-backed local CSV/Parquet querying and JSON prediction payload loading in the backend

---

## Model & Training

### Model Components

- **Backbone**: CEEMDAN-Transformer-DNN
- **Meta-learning**: BMAML-SVGD-style few-shot adaptation
- **Uncertainty**: particle-style prediction distribution
- **Tuning**: Ray Tune + ASHA scheduler
- **Checkpoint**: `nasa_bmaml_best_re.pt`

### Experimental Setting

```text
Support Set: 16 cycles
Query Set: 16 cycles
Meta-Train Batteries: 14 batteries
Meta-Val Batteries: 2 batteries
Meta-Test Batteries: 6 batteries
Evaluation Ratio: r_ratio = 0.20
```

### Training Results

| Metric | Value | Description |
|---|---:|---|
| RMSE | 7.46 cycles | Query set prediction error at r_ratio = 0.20 |
| MAE | 6.82 cycles | Mean absolute error at r_ratio = 0.20 |

> Note: These results validate the implemented inference pipeline under a fixed experimental setting. They are not presented as a state-of-the-art benchmark claim.

---

## Engineering Challenges Solved

### 1. Local vs Docker Path Mismatch

**Problem**: Model checkpoints and data files worked locally but failed inside Docker/Hugging Face Spaces due to different runtime paths.

**Solution**: Added deployment-aware path handling through backend settings and runtime entrypoints.

### 2. CPU-Based Live Reinference Runtime

**Problem**: Live reinference takes around 40 seconds in a CPU-only Hugging Face Spaces environment.

**Solution**: Combined precomputed JSON cache for fast initial loading with on-demand live reinference for detailed analysis.

### 3. Frontend-Backend State Synchronization

**Problem**: Reinference results needed to update prediction curves, uncertainty bands, confidence values, metrics, baseline restoration, and dashboard state consistently.

**Solution**: Designed API response payloads and React state flow to synchronize prediction, standard deviation, confidence, and metrics.

### 4. Backend 500 Errors and Deployment Stability

**Problem**: Missing files, path mismatches, or unavailable precomputed data could break the dashboard flow.

**Solution**: Added fallback behavior, diagnostics, and deployment-aware file handling to improve runtime stability.

### 5. Model Output to User-Facing Dashboard

**Problem**: Raw model outputs are difficult to interpret directly.

**Solution**: Converted predictions into visual curves, uncertainty bands, degradation indicators, SHAP feature importance, and frontend CSV download for the currently displayed prediction curve.

---

## Backend API

The backend is implemented with FastAPI and provides inference, degradation monitoring, and explainability-related endpoints.

### Main Endpoints

```text
GET  /api/battery/{id}/precomputed
POST /api/battery/{id}/reinfer?r_ratio=X
GET  /api/battery/{id}/degradation-monitoring
GET  /api/fixed4/shap-current
```

### Backend Responsibilities

- Load precomputed prediction payloads.
- Run live model reinference.
- Query local CSV/Parquet feature data through DuckDB.
- Load JSON prediction payloads for precomputed inference results.
- Generate degradation monitoring outputs.
- Provide SHAP-based feature importance results.

---

## Frontend Dashboard

The frontend is implemented with React, Vite, TailwindCSS, and Plotly.js.

### Main Views

- **Overview**: RUL prediction curve, support/query split, uncertainty band, and metrics.
- **Degradation**: capacity, DCR, impedance, temperature, current stress, and proxy degradation signals.
- **Compare**: multi-battery comparison view.
- **Explainability**: uncertainty summary and SHAP-based feature importance.
- **Live Reinference**: user-triggered inference flow with updated dashboard state.

---

## Tech Stack

| Area | Technology |
|---|---|
| Deep Learning | PyTorch |
| Meta-Learning | BMAML-SVGD-style few-shot learning |
| Sequence Modeling | CEEMDAN + Transformer + DNN |
| Signal Processing | CEEMDAN, robust scaling |
| Feature Engineering | 40 time-series degradation features |
| Hyperparameter Tuning | Ray Tune, ASHA scheduler |
| Backend | FastAPI, DuckDB-backed local feature data access with CSV/Parquet support |
| Frontend | React, Vite, TailwindCSS, Plotly.js |
| Deployment | Docker, Hugging Face Spaces, Git LFS |
| Explainability | SHAP-based feature importance |

---

## Project Structure

```text
battery-rul-dashboard/
├─ deep_learning/
│  └─ core/
│     ├─ train_meta.py           # BMAML-style meta-training
│     ├─ models.py               # CEEMDAN-Transformer-DNN model components
│     ├─ meta_utils.py           # Inner loop and SVGD utilities
│     ├─ bmaml_runtime.py        # Inference runtime entrypoint
│     ├─ feature_shap_bmaml.py   # SHAP-based feature importance
│     └─ config.py               # Hyperparameters and configuration
│
├─ backend/
│  └─ app/
│     ├─ main.py                 # FastAPI routes and inference APIs
│     ├─ data_access.py          # DuckDB-backed local feature data access utilities
│     ├─ settings.py             # Deployment-aware configuration
│     └─ diagnostics.py          # Degradation monitoring logic
│
├─ frontend/
│  └─ src/
│     └─ ui/
│        ├─ App.tsx              # Main dashboard UI and state flow
│        └─ ExplainabilityAnomalyV30.tsx
│
├─ data/
│  ├─ nasa_features_rul.csv      # Engineered feature data
│  ├─ precomputed/               # Cached prediction payloads
│  └─ precomputed_from_export_v2/
│
├─ scripts/
│  └─ prefix_inference_viz_meta_restored_v3.py
│
├─ core_checkpoints/
│  └─ nasa_bmaml_best_re.pt      # Final model checkpoint, tracked with Git LFS
│
├─ Dockerfile                    # Full-stack Docker build
├─ hf_app.py                     # FastAPI + React serving entrypoint
├─ run_bmaml_reinfer.py          # Live reinference wrapper
└─ README.md
```

---

## Local Development

### 1. Create Environment

```bash
conda create -n battery-maml python=3.10
conda activate battery-maml
```

### 2. Install Backend

```bash
pip install -e ./backend
```

### 3. Install ML Dependencies

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install ray[tune]==2.20.0
pip install pyyaml
```

### 4. Install Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Model Training

```bash
cd deep_learning/core
python train_meta.py --config config.yaml \
  --k_shot 30 \
  --q_query 128 \
  --meta_lr 0.001 \
  --inner_lr 0.01
```

---

## Live Inference

```bash
python run_bmaml_reinfer.py \
  --battery B0043 \
  --checkpoint ../core_checkpoints/nasa_bmaml_best_re.pt \
  --r_ratio 0.2
```

---

## Deployment

This project is deployed on Hugging Face Spaces using Docker.

### Deployment Notes

- **Docker**: builds and serves the full-stack app.
- **Git LFS**: tracks model checkpoint files.
- **Dynamic Paths**: handles local vs Docker/HF runtime paths.
- **Precomputed Cache**: improves initial dashboard loading speed.
- **Live Reinference Timeout**: configured to allow long CPU-based inference requests.

### Push to Hugging Face Spaces

```bash
git push hf main
```

---

## Limitations & Future Work

| Area | Current Status | Future Improvement |
|---|---|---|
| Dataset | NASA battery dataset only | Add CALCE or real-world EV battery data |
| Inference | CPU-based live reinference | GPU-backed cloud inference or optimized runtime |
| Monitoring | Basic diagnostics and degradation indicators | Structured logging and model monitoring |
| Experiment Tracking | Limited experiment metadata | Integrate W&B / MLflow consistently |
| CI/CD | Manual deployment flow | Add automated tests and deployment pipeline |
| Testing | Manual validation | Add API tests, inference regression tests, and frontend flow tests |
| Code Structure | Large backend/frontend modules | Modular FastAPI routers and reusable React components |
| Model Optimization | Full checkpoint inference | Quantization or distilled inference model |

---

## Methodological Background

This project is methodologically inspired by MAML, Bayesian meta-learning, SVGD, and CEEMDAN-based signal decomposition.

Rather than claiming a full reproduction of each original method, the implementation uses BMAML-SVGD-style few-shot adaptation and uncertainty estimation on top of CEEMDAN-Transformer-DNN degradation modeling.

- Finn et al. (2017), Model-Agnostic Meta-Learning (MAML)  
  Used as the conceptual basis for few-shot adaptation.

- Bayesian MAML / Bayesian meta-learning  
  Used as the motivation for uncertainty-aware meta-learning and particle-based prediction.

- Liu & Wang (2016), Stein Variational Gradient Descent (SVGD)  
  Used as the basis for particle-style Bayesian uncertainty estimation.

- Torres et al. (2011), CEEMDAN  
  Used as the signal decomposition background for separating local capacity fluctuation/regeneration components from long-term degradation trends.

---

## License

MIT License

---

## Author

**onekindalphal** — Full-stack AI / Deep Learning Engineer

- Data pipeline: feature engineering, few-shot task construction, robust scaling
- Deep learning: BMAML-SVGD-style adaptation, CEEMDAN-Transformer-DNN backbone, uncertainty estimation
- Backend: FastAPI, DuckDB-backed local feature data access with CSV/Parquet support, inference APIs, degradation monitoring
- Frontend: React dashboard, visualization, live reinference state management
- Deployment: Docker, Hugging Face Spaces, Git LFS

---

## Links

- **Live Demo**: https://onekindalpha-battery-rul-dashboard-bmaml-svgd.hf.space
