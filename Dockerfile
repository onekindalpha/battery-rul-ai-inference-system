FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/backend
ENV PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    nodejs \
    npm \
  && rm -rf /var/lib/apt/lists/*

COPY backend ./backend
COPY frontend ./frontend
COPY data ./data
COPY deep_learning ./deep_learning
COPY scripts ./scripts
COPY core_checkpoints ./core_checkpoints
COPY hf_app.py ./hf_app.py
COPY run_bmaml_reinfer.py ./run_bmaml_reinfer.py
COPY export_rul_dashboard_data_meta_fixed.py ./export_rul_dashboard_data_meta_fixed.py

RUN pip install --upgrade pip \
  && pip install -e ./backend

# Make deployment paths compatible with backend code that may resolve data
# relative to either the repo root or the backend package.
RUN ln -sfn /app/data /app/backend/data \
  && ln -sfn /app/data /app/backend/app/data \
  && ln -sfn /app/core_checkpoints /app/backend/core_checkpoints \
  && ln -sfn /app/deep_learning /app/backend/deep_learning

RUN cd frontend \
  && npm ci \
  && npx vite build

EXPOSE 7860

COPY hf_front_app.py ./hf_front_app.py
COPY requirements-hf-reinfer.txt ./requirements-hf-reinfer.txt
RUN pip install --no-cache-dir --force-reinstall -r requirements-hf-reinfer.txt && pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
CMD ["python", "-m", "uvicorn", "hf_front_app:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "900"]
