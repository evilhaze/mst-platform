# MST Platform

Unified ML platform combining three microservices into a single FastAPI application:

- **ROI/CTR Prediction** — real-time ad conversion prediction with Redis caching
- **Creative Analysis** — AI-powered ad text feature extraction, classification, and improvement tips
- **Creative Generation** — few-shot ad variant generation scored by a trained classifier

## Architecture

```
                         +-------------------+
                         |   MST Platform    |
                         |   FastAPI :8000   |
                         +--------+----------+
                                  |
                 +----------------+----------------+
                 |                |                 |
          +------+------+  +-----+-----+  +-------+-------+
          | Redis :6379 |  | MLflow    |  | Anthropic API |
          | (cache)     |  | :5000     |  | (Claude)      |
          +-------------+  +-----------+  +---------------+
```

## Quick Start

```bash
git clone https://github.com/evilhaze/mst-platform
cd mst-platform
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
docker compose up -d
# API available at http://localhost:8000 after ~60 seconds
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Single ROI/CTR prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 10K items) |
| `POST` | `/creatives/analyze` | Analyze ad creative text |
| `POST` | `/creatives/generate` | Generate ad creative variants |
| `GET` | `/health` | Service health check |
| `GET` | `/metrics` | Model quality metrics |

## curl Examples

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "impressions": 50000,
      "clicks": 1200,
      "spend": 850.0,
      "bid_amount": 0.75,
      "hour_of_day": 14,
      "day_of_week": 2,
      "campaign_age_days": 45,
      "geo": "US",
      "device_type": "desktop",
      "ad_format": "native",
      "placement": "in-feed"
    }
  }'
```

### POST /predict/batch

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "impressions": 50000,
        "clicks": 1200,
        "spend": 850.0,
        "bid_amount": 0.75,
        "hour_of_day": 14,
        "day_of_week": 2,
        "campaign_age_days": 45,
        "geo": "US",
        "device_type": "desktop",
        "ad_format": "native",
        "placement": "in-feed"
      },
      {
        "impressions": 10000,
        "clicks": 300,
        "spend": 200.0,
        "bid_amount": 0.50,
        "hour_of_day": 9,
        "day_of_week": 5,
        "campaign_age_days": 10,
        "geo": "UK",
        "device_type": "mobile",
        "ad_format": "video",
        "placement": "top"
      }
    ]
  }'
```

### POST /creatives/analyze

```bash
curl -X POST http://localhost:8000/creatives/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Только сегодня! Бонус 500% на первый депозит — забери свой выигрыш прямо сейчас!",
    "vertical": "gambling"
  }'
```

### POST /creatives/generate

```bash
curl -X POST http://localhost:8000/creatives/generate \
  -H "Content-Type: application/json" \
  -d '{
    "original_text": "Играй и выигрывай в лучшем казино",
    "offer": "500% бонус на депозит",
    "geo": "RU",
    "vertical": "casino",
    "n_variants": 5
  }'
```

### GET /health

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "checks": {
    "redis": "ok",
    "mlflow": "ok",
    "model": "ok",
    "creative_analyzer": "ok",
    "creative_generator": "ok"
  },
  "cache": {
    "hit_rate": 0.42,
    "hits": 1054,
    "misses": 1456,
    "errors": 0
  }
}
```

### GET /metrics

```bash
curl http://localhost:8000/metrics
```

Response:
```json
{
  "roi_predictor": {
    "model_version": "1.0.0",
    "trained_at": "2025-03-15T10:30:00",
    "roc_auc": 0.8734,
    "threshold": 0.5,
    "total_predictions": 2510
  },
  "creative_classifier": {
    "roc_auc": 0.9102,
    "model_type": "random_forest",
    "cv_results": {
      "logistic": 0.8845,
      "random_forest": 0.9102
    }
  }
}
```

## Authentication

Set `ALLOWED_API_KEYS` in `.env` to enable auth (comma-separated):

```env
ALLOWED_API_KEYS=key1-abc,key2-xyz
```

Then pass the key in every request:

```bash
curl -H "X-API-Key: key1-abc" http://localhost:8000/health
```

If `ALLOWED_API_KEYS` is empty or unset, the API runs in **dev mode** (no auth required).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Claude API key for creative analysis/generation |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `MLFLOW_URL` | — | MLflow tracking server URL |
| `MODEL_PATH` | `models/model.pkl` | Local fallback model path |
| `ALLOWED_API_KEYS` | — | Comma-separated API keys (empty = dev mode) |
| `LOG_LEVEL` | `info` | Logging level |
| `CACHE_TTL_SECONDS` | `300` | Redis cache TTL for predictions |
