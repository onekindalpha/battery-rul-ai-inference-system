# Development Notes

This file contains local development, deployment commands, runtime notes, and implementation references for Battery RUL AI Inference System.

## Local Development

### Backend

```bash
cd backend
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Live Reinference Example

```bash
python run_bmaml_reinfer.py \
  --battery B0043 \
  --checkpoint ../core_checkpoints/nasa_bmaml_best_re.pt \
  --r_ratio 0.2
```

Paths may vary depending on local checkpoint and data locations. The public demo uses prepared assets and precomputed prediction files for stable dashboard loading.

## Deployment

This project is deployed on Hugging Face Spaces using Docker.

Deployment notes:

- Docker builds and serves the full-stack app.
- Git LFS tracks model checkpoint files.
- Precomputed cache improves initial dashboard loading speed.
- Live reinference is available as an on-demand path in the demo environment.

```bash
git push hf main
```

## Runtime Notes

The dashboard is designed to prefer precomputed prediction results for fast demo loading. Live reinference is available as a slower path when checkpoint and feature artifacts are available.

## Related Repository

- Battery Technical Document RAG Assistant: https://github.com/onekindalpha/battery-technical-document-rag
