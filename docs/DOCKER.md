# Docker Guide

## Quick Start

**Requirements:** Docker 24+, Compose v2, Buildx plugin

**Setup:**
```bash
make wizard  # Interactive (recommended)
# OR
cp .env.docker.example .env && make docker-up
```

Access at http://localhost:8080

## Architecture

Files: `docker-compose.yml` (main), `.cuda.yml` (GPU), `.dev.yml` (dev), `.prod.yml` (production), `Dockerfile` (multi-stage)

## Services

**Data:** Qdrant (6333, 6334), PostgreSQL (5432), Redis (6379)
**App:** WebUI (8080), Vecpipe (8000), Worker, Beat, Flower (5555, profile: backend)
**Test:** PostgreSQL Test (55432, profile: testing)

All have health checks and resource limits.

## Periodic tasks (Celery Beat)

The `beat` service schedules periodic Celery tasks (including the continuous source sync dispatcher). If `beat` is not running, continuous sync sources will not auto-run.

## Profiles

- **Default**: Core services (`docker compose up -d`)
- **backend**: + Flower (`--profile backend`)
- **testing**: + Test PostgreSQL (`--profile testing`)

## Volumes

**Named:** `qdrant_storage`, `postgres_data`, `redis_data`, `postgres_test_data`

**Bind mounts:** `./data` (operations), `./models` (HF cache), `./logs`, `${DOCUMENT_PATH}` (read-only)

## GPU

Use `docker-compose.cuda.yml` for INT8 quantization:
```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d
```

Config: `CUDA_VISIBLE_DEVICES=0`, `DEFAULT_QUANTIZATION=float16`

Memory: Qwen3-0.6B uses ~1.2GB (float16), ~0.6GB (int8)

## Configuration

**Required credentials (v7.1+):**
- `JWT_SECRET_KEY` - Authentication secret
- `POSTGRES_PASSWORD` - Database password
- `REDIS_PASSWORD` - Redis authentication
- `QDRANT_API_KEY` - Vector database authentication
- `FLOWER_USERNAME`, `FLOWER_PASSWORD` - Task monitoring UI

Docker Compose enforces these values on startup - containers won't start with placeholder values. Run `make wizard` to auto-generate all credentials, or generate manually with `openssl rand -hex 32`.

**Optional:** `CONNECTOR_SECRETS_KEY` - Fernet key for encrypting connector credentials (Git/IMAP/etc.). Set to empty to disable.

See `.env.docker.example` for full list.

## Networking

All services on `semantik-network`. Internal DNS: `postgres:5432`, `qdrant:6333`, etc.

## Security

Containers: `no-new-privileges`, dropped capabilities, non-root (UID/GID in `.env`)

Generate secrets: `openssl rand -hex 32`

## Commands

```bash
make docker-up           # Start
make docker-down         # Stop
make docker-logs         # Logs
docker compose logs -f webui  # Service logs
docker compose exec webui /bin/bash  # Shell
docker stats             # Resource usage
make docker-build-fresh  # Rebuild
docker system prune -a   # Cleanup
```

## Development

Dev mode: `make docker-dev-up` (live reload, debug logging)

Tests: `docker compose run --rm webui pytest`

## Production

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

Create your own `nginx.conf` for reverse proxy. Production config expects `./nginx.conf` and `./ssl/` directory.

## Troubleshooting

**Won't start:** `docker compose logs <service>`, check ports with `lsof`
**Permissions:** `sudo chown -R 1000:1000 ./data ./models ./logs`
**Memory:** `docker stats`, reduce quantization to int8
**GPU:** Install `nvidia-container-toolkit`, restart Docker
**DB issues:** `docker compose logs postgres`, test with `psql -U semantik`

## Monitoring

Metrics: `docker compose exec webui curl http://localhost:9091/metrics` (internal)

Flower: `docker compose --profile backend up -d`, then http://localhost:5555
