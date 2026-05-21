# Battery RUL AI Inference System

배터리 잔여수명(RUL, Remaining Useful Life) 예측, live reinference, 열화 모니터링, 설명 가능성 분석을 제공하는 딥러닝 기반 full-stack AI inference system입니다.

**목표**: 초기 cycle 관측값으로 배터리 RUL을 예측하고, 예측 결과와 열화 신호를 웹 대시보드에서 확인할 수 있는 end-to-end AI inference system을 구현하는 것입니다.

**Live Demo**: https://onekindalpha-battery-rul-dashboard-bmaml-svgd.hf.space  
**Demo Video**: https://github.com/user-attachments/assets/1e05d64d-b9e3-47ac-abc4-06048ae3b7a7

데모 영상은 cycle 단위 RUL 추론, 열화 모니터링, 배터리 비교, EOL 이전 anomaly evidence 기반 설명 가능성 분석, SHAP 기반 feature importance, live reinference 흐름을 보여줍니다.

---

## TL;DR

- NASA battery 데이터를 기반으로 RUL 예측과 열화 모니터링을 제공하는 end-to-end deep learning inference system을 구현했습니다.
- 배터리 cycle 데이터에서 domain-informed time-series feature를 설계하고, few-shot support/query task를 구성했습니다.
- CEEMDAN-Transformer-DNN backbone 위에 BMAML-SVGD-style uncertainty-aware meta-learning 구조를 적용했습니다.
- FastAPI + React 대시보드를 구축하고, live reinference, uncertainty visualization, degradation monitoring, SHAP-based explainability, precomputed baseline restoration, CSV download 기능을 배포했습니다.
- DuckDB-backed local feature data access와 CSV/Parquet, local JSON prediction payload를 활용해 dashboard data loading과 precomputed inference result를 지원했습니다.

---

## Project Overview

이 프로젝트는 battery domain feature, sequence modeling, inference API, 배포 가능한 웹 대시보드를 하나로 연결한 applied deep learning system입니다.

- **Problem**: 초기 cycle 관측값으로 배터리 잔여수명을 예측하고, 열화 양상을 모니터링합니다.
- **Data Pipeline**: NASA battery cycle data → domain-informed feature engineering → few-shot support/query task construction.
- **Modeling**: CEEMDAN-Transformer-DNN backbone 기반 BMAML-SVGD-style few-shot Bayesian meta-learning.
- **Inference System**: precomputed cache와 optional live reinference를 지원하는 PyTorch model inference.
- **Application**: FastAPI backend + React/Vite dashboard를 Docker 기반 Hugging Face Spaces에 배포.

---

## My Role

데이터 전처리부터 모델 추론, 대시보드 개발, 배포까지 핵심 pipeline을 구현했습니다.

- **Data / Feature Engineering**: NASA battery preprocessing, 40개 time-series feature, support/query task construction, robust scaling.
- **Deep Learning**: BMAML-SVGD-style meta-learning, CEEMDAN-Transformer-DNN backbone, uncertainty estimation.
- **Backend**: FastAPI inference API, DuckDB-backed local feature data access, degradation monitoring endpoint, precomputed/live inference result handling.
- **Frontend**: React dashboard, RUL curve visualization, uncertainty band rendering, degradation tab, live reinference state management.
- **Deployment**: Docker 기반 full-stack app deployment on Hugging Face Spaces, Git LFS checkpoint handling.

---

## Key Features

### RUL Prediction Dashboard

- support period부터 query period까지 predicted RUL trajectory를 시각화합니다.
- prediction curve, ground-truth reference, model uncertainty band를 함께 표시합니다.
- 선택한 battery와 observation ratio 기준으로 RMSE, MAE, confidence-related metrics를 계산합니다.

### Live Reinference

- dashboard에서 직접 실행 가능한 `Initialize & Reinference` flow를 제공합니다.
- Hugging Face Spaces의 CPU 환경에서 on-demand model inference를 실행합니다.
- reinference 이후 prediction curve, uncertainty value, confidence metric, dashboard state를 업데이트합니다.
- live reinference 이후 precomputed baseline result로 복귀할 수 있습니다.
- 현재 표시 중인 prediction curve를 CSV file로 다운로드할 수 있습니다.

### Degradation Monitoring

- capacity, DCR, impedance, temperature, current stress, LLI proxy signal 등 열화 관련 signal을 모니터링합니다.
- cycle-level evidence를 활용해 abnormal degradation behavior를 확인합니다.
- 단일 RUL 숫자만 보는 것이 아니라, 열화 중심의 추가 해석 화면을 제공합니다.

### Explainability & Uncertainty

- 여러 SVGD-style prediction particle을 활용해 prediction uncertainty를 추정합니다.
- SHAP-based analysis를 통해 global feature importance를 시각화합니다.
- model output과 degradation-related feature behavior를 함께 확인할 수 있도록 구성했습니다.

### Deployment-Oriented Inference Flow

- 빠른 초기 로딩을 위해 precomputed prediction cache를 사용합니다.
- deeper analysis가 필요할 때 live reinference를 실행할 수 있습니다.
- local 환경과 Docker/Hugging Face runtime path 차이를 처리할 수 있도록 deployment-aware configuration을 적용했습니다.

---

## System Architecture

![System architecture diagram](docs/assets/system_architecture_v6.svg)

배포된 dashboard는 두 가지 inference path를 중심으로 구성됩니다. fast path는 선택한 battery와 observation ratio에 대해 precomputed JSON payload를 로드합니다. optional live path는 PyTorch reinference wrapper를 실행하고, live prediction과 uncertainty value로 dashboard를 업데이트합니다.

같은 FastAPI backend는 degradation monitoring, SHAP feature importance, CSV export, DuckDB-backed feature access를 제공합니다. React frontend는 prediction, monitoring, comparison, explainability, playback, baseline restore, export flow를 렌더링합니다.

---

## Deep Learning Approach

### Few-Shot RUL Prediction

실제 배터리 운용 환경에서는 배터리 수명 초기에 장기간 열화 데이터가 충분히 확보되지 않을 수 있습니다. 이 프로젝트는 early-cycle RUL estimation을 few-shot prediction 문제로 구성했습니다.

- **Support set**: 초기 관측 cycle
- **Query set**: 예측 대상 future cycle
- **Task**: 제한된 early-cycle information으로 각 battery에 adaptation

### BMAML-SVGD-Style Meta-Learning

![Model flow diagram](docs/assets/model_flow_v6.svg)

모델은 Bayesian MAML과 SVGD-based uncertainty estimation에서 영감을 받은 구조입니다. sequence feature와 summary feature, CEEMDAN-based decomposition feature, Transformer encoder, DNN prediction head, SVGD-style particle을 활용해 uncertainty-aware RUL prediction을 수행합니다.

### Why This Approach

- **Few-shot adaptation**: 제한된 early-cycle observation으로 새로운 battery에 적응할 수 있습니다.
- **Sequence modeling**: 시간에 따른 degradation pattern을 포착합니다.
- **Uncertainty estimation**: 단일 point prediction보다 더 많은 정보를 제공합니다.
- **Domain-informed features**: ML prediction과 물리적 degradation behavior를 연결하는 데 도움을 줍니다.

---

## Data Pipeline

![Data pipeline diagram](docs/assets/data_pipeline_v6.svg)

Raw battery measurement는 degradation-related time-series feature로 변환됩니다. Feature에는 capacity degradation velocity, DCR/impedance growth rate, temperature stress indicator, current/load stress metric, CEEMDAN-based IMF decomposition feature, LLI/LAM proxy signal 등이 포함됩니다.

Main data components:

- **Feature count**: 40개 time-series feature
- **Scaling**: sequence data를 위한 custom robust 3D scaling
- **Task setup**: meta-learning을 위한 support/query construction
- **Data access**: backend에서 DuckDB-backed local CSV/Parquet querying과 JSON prediction payload loading

---

## Model & Training

### Model Components

- **Backbone**: CEEMDAN-Transformer-DNN
- **Meta-learning**: BMAML-SVGD-style few-shot adaptation
- **Uncertainty**: particle-style prediction distribution
- **Tuning**: Ray Tune + ASHA scheduler
- **Checkpoint**: `nasa_bmaml_best_re.pt`

### Experimental Setting

- **Support Set**: 16 cycles
- **Query Set**: 16 cycles
- **Meta-Train Batteries**: 14 batteries
- **Meta-Val Batteries**: 2 batteries
- **Meta-Test Batteries**: 6 batteries
- **Evaluation Ratio**: r_ratio = 0.20

### Training Results

- **RMSE**: 7.46 cycles at r_ratio = 0.20
- **MAE**: 6.82 cycles at r_ratio = 0.20

> 이 결과는 고정된 실험 설정에서 구현한 inference pipeline의 결과를 보여주기 위한 값입니다. state-of-the-art benchmark claim으로 제시하는 것은 아닙니다.

---

## Engineering Challenges Solved

### Local vs Docker Path Mismatch

**Problem**: local에서는 model checkpoint와 data file이 정상 동작했지만, Docker/Hugging Face Spaces 환경에서는 runtime path 차이로 실패할 수 있었습니다.

**Solution**: backend settings와 runtime entrypoint에 deployment-aware path handling을 추가했습니다.

### CPU-Based Live Reinference Runtime

**Problem**: CPU-only Hugging Face Spaces 환경에서는 live reinference가 약 40초 정도 걸릴 수 있습니다.

**Solution**: 빠른 초기 로딩을 위해 precomputed JSON cache를 사용하고, 상세 분석이 필요할 때 on-demand live reinference를 실행하는 구조로 구성했습니다.

### Frontend-Backend State Synchronization

**Problem**: reinference result가 prediction curve, uncertainty band, confidence value, metric, baseline restoration, dashboard state를 일관되게 업데이트해야 했습니다.

**Solution**: API response payload와 React state flow를 설계해 prediction, standard deviation, confidence, metrics가 함께 동기화되도록 했습니다.

### Backend 500 Errors and Deployment Stability

**Problem**: missing files, path mismatch, unavailable precomputed data 등이 dashboard flow를 깨뜨릴 수 있었습니다.

**Solution**: fallback behavior, diagnostics, deployment-aware file handling을 추가해 runtime stability를 개선했습니다.

### Model Output to User-Facing Dashboard

**Problem**: raw model output은 사용자가 직접 해석하기 어렵습니다.

**Solution**: prediction을 visual curve, uncertainty band, degradation indicator, SHAP feature importance, frontend CSV download로 변환했습니다.

---

## Backend API

![Backend API flow diagram](docs/assets/backend_api_flow_v6.svg)

Backend는 FastAPI로 구현되었으며 inference, degradation monitoring, explainability, export endpoint를 제공합니다.

Main endpoint groups:

- precomputed prediction loading
- live reinference
- degradation monitoring
- SHAP-based feature importance
- current prediction result에 대한 CSV export

Backend responsibilities:

- precomputed prediction payload loading
- live model reinference 실행
- DuckDB를 통한 local CSV/Parquet feature data query
- precomputed inference result용 JSON prediction payload loading
- degradation monitoring output 생성
- SHAP-based feature importance result 제공

---

## Frontend Dashboard

Frontend는 React, Vite, TailwindCSS, Plotly.js로 구현했습니다.

Main views:

- **Overview**: RUL prediction curve, support/query split, uncertainty band, metrics.
- **Degradation**: capacity, DCR, impedance, temperature, current stress, proxy degradation signal.
- **Compare**: multi-battery comparison view.
- **Explainability**: uncertainty summary, cumulative anomaly evidence, SHAP-based feature importance, model architecture.
- **Live Reinference**: user-triggered inference flow와 updated dashboard state.

---

## Tech Stack

- **Deep Learning**: PyTorch
- **Meta-Learning**: BMAML-SVGD-style few-shot learning
- **Sequence Modeling**: CEEMDAN + Transformer + DNN
- **Signal Processing**: CEEMDAN, robust scaling
- **Feature Engineering**: 40 time-series degradation features
- **Hyperparameter Tuning**: Ray Tune, ASHA scheduler
- **Backend**: FastAPI, DuckDB-backed local feature data access with CSV/Parquet support
- **Frontend**: React, Vite, TailwindCSS, Plotly.js
- **Deployment**: Docker, Hugging Face Spaces, Git LFS
- **Explainability**: SHAP-based feature importance

---

## Repository Map

![Repository map diagram](docs/assets/repository_map_v6.svg)

위 diagram은 repository tree 전체를 길게 나열하는 대신, deployed dashboard에서 중요한 구현 진입점을 요약합니다.

---

## Local Development

### Create Environment

```bash
conda create -n battery-maml python=3.10
conda activate battery-maml
```

### Install Backend

```bash
pip install -e ./backend
```

### Install ML Dependencies

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install ray[tune]==2.20.0
pip install pyyaml
```

### Install Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Model Training

```bash
cd deep_learning/core
python train_meta.py --config config.yaml   --k_shot 30   --q_query 128   --meta_lr 0.001   --inner_lr 0.01
```

---

## Live Inference

```bash
python run_bmaml_reinfer.py   --battery B0043   --checkpoint ../core_checkpoints/nasa_bmaml_best_re.pt   --r_ratio 0.2
```

---

## Deployment

This project is deployed on Hugging Face Spaces using Docker.

Deployment notes:

- **Docker**: full-stack app을 build하고 serving합니다.
- **Git LFS**: model checkpoint file을 추적합니다.
- **Dynamic Paths**: local vs Docker/HF runtime path를 처리합니다.
- **Precomputed Cache**: initial dashboard loading speed를 개선합니다.
- **Live Reinference Timeout**: CPU-based inference request를 처리할 수 있도록 설정했습니다.

```bash
git push hf main
```

---

## Limitations & Future Work

- **Dataset**: 현재 NASA battery dataset 중심입니다. 향후 CALCE 또는 real-world EV battery data를 추가할 수 있습니다.
- **Inference**: deployed demo의 live reinference는 CPU-based입니다. 향후 GPU-backed cloud inference 또는 optimized runtime을 적용할 수 있습니다.
- **Monitoring**: 현재 diagnostics는 dashboard-oriented입니다. 향후 structured logging과 model monitoring을 추가할 수 있습니다.
- **Experiment Tracking**: experiment metadata가 제한적입니다. 향후 W&B 또는 MLflow를 일관되게 연동할 수 있습니다.
- **Testing**: 현재 validation은 대부분 manual입니다. 향후 API test, inference regression test, frontend flow test를 추가할 수 있습니다.
- **Code Structure**: 일부 backend/frontend module이 큽니다. 향후 FastAPI router와 React component를 더 modular하게 분리할 수 있습니다.
- **Model Optimization**: 현재 inference는 full checkpoint를 사용합니다. 향후 quantization 또는 distilled inference model을 검토할 수 있습니다.

---

## Methodological Background

이 프로젝트는 MAML, Bayesian meta-learning, SVGD, CEEMDAN-based signal decomposition에서 방법론적 영감을 받았습니다.

각 원 방법론을 완전 재현했다고 주장하기보다는, CEEMDAN-Transformer-DNN degradation modeling 위에 BMAML-SVGD-style few-shot adaptation과 uncertainty estimation을 적용한 구현입니다.

- Finn et al. (2017), Model-Agnostic Meta-Learning (MAML): few-shot adaptation의 개념적 기반으로 사용했습니다.
- Bayesian MAML / Bayesian meta-learning: uncertainty-aware meta-learning과 particle-based prediction의 동기로 사용했습니다.
- Liu & Wang (2016), Stein Variational Gradient Descent (SVGD): particle-style Bayesian uncertainty estimation의 기반으로 사용했습니다.
- Torres et al. (2011), CEEMDAN: local capacity fluctuation/regeneration component와 long-term degradation trend를 분리하기 위한 signal decomposition 배경으로 사용했습니다.

---

## License

MIT License

---

## Author

**onekindalpha** — Full-stack AI / Deep Learning Engineer

- Data pipeline: feature engineering, few-shot task construction, robust scaling
- Deep learning: BMAML-SVGD-style adaptation, CEEMDAN-Transformer-DNN backbone, uncertainty estimation
- Backend: FastAPI, DuckDB-backed local feature data access with CSV/Parquet support, inference APIs, degradation monitoring
- Frontend: React dashboard, visualization, live reinference state management
- Deployment: Docker, Hugging Face Spaces, Git LFS

---

## Links

- **Live Demo**: https://onekindalpha-battery-rul-dashboard-bmaml-svgd.hf.space
- **Demo Video**: https://github.com/user-attachments/assets/1e05d64d-b9e3-47ac-abc4-06048ae3b7a7
