# Makefile for Document Embedding System

.PHONY: help install dev-install format lint type-check test test-coverage clean
.PHONY: frontend-install frontend-build frontend-dev frontend-test build dev
.PHONY: docker-up docker-down docker-logs docker-build-fresh docker-ps docker-restart
.PHONY: docker-postgres-up docker-postgres-down docker-postgres-logs docker-shell-postgres
.PHONY: docker-postgres-backup docker-postgres-restore
.PHONY: dev-local docker-dev-up docker-dev-down docker-dev-logs

help:
	@echo "Available commands:"
	@echo "  install        Install production dependencies"
	@echo "  dev-install    Install development dependencies"
	@echo "  format         Format code with black and isort"
	@echo "  lint           Run linting with ruff"
	@echo "  type-check     Run type checking with mypy"
	@echo "  test           Run all tests"
	@echo "  test-ci        Run tests excluding E2E (for CI)"
	@echo "  test-e2e       Run only E2E tests (requires running services)"
	@echo "  test-coverage  Run tests with coverage report"
	@echo "  clean          Clean up generated files"
	@echo ""
	@echo "Docker commands:"
	@echo "  wizard            Interactive Docker setup wizard (TUI)"
	@echo "  docker-up         Start all services with PostgreSQL"
	@echo "  docker-down       Stop and remove all containers"
	@echo "  docker-logs       View logs from all services"
	@echo "  docker-build-fresh Rebuild images without cache"
	@echo "  docker-ps         Show status of all containers"
	@echo "  docker-restart    Restart all services"
	@echo ""
	@echo "PostgreSQL specific commands:"
	@echo "  docker-postgres-logs   View PostgreSQL container logs"
	@echo "  docker-shell-postgres  Access PostgreSQL shell"
	@echo "  docker-postgres-backup Create database backup"
	@echo "  docker-postgres-restore Restore database from backup"
	@echo ""
	@echo "Frontend commands:"
	@echo "  frontend-install  Install frontend dependencies"
	@echo "  frontend-build    Build frontend for production"
	@echo "  frontend-dev      Start frontend dev server"
	@echo "  frontend-test     Run frontend tests"
	@echo ""
	@echo "Integrated commands:"
	@echo "  build          Build entire project"
	@echo "  dev            Start development environment"
	@echo ""
	@echo "Local development (webui runs locally with hot reload):"
	@echo "  dev-local      Start webui locally + services in Docker"
	@echo "  docker-dev-up  Start only supporting services in Docker"
	@echo "  docker-dev-down Stop Docker development services"
	@echo "  docker-dev-logs View logs from development services"

install:
	poetry install --no-dev

dev-install:
	poetry install

format:
	poetry run black packages/vecpipe packages/webui packages/shared tests
	poetry run isort packages/vecpipe packages/webui packages/shared tests

lint:
	poetry run ruff check packages/vecpipe packages/webui packages/shared tests

type-check:
	poetry run mypy packages/vecpipe packages/webui packages/shared --ignore-missing-imports

test:
	poetry run pytest tests -v

test-ci:
	poetry run pytest tests -v -m "not e2e"

test-e2e:
	poetry run pytest tests -v -m e2e

test-coverage:
	poetry run pytest tests -v --cov=packages.vecpipe --cov=packages.webui --cov=packages.shared --cov-report=html --cov-report=term

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov .pytest_cache .mypy_cache .ruff_cache
	rm -rf *.egg-info dist build

# Docker commands for the new setup
# Interactive wizard for setting up Semantik with Docker
# - Configures GPU/CPU deployment
# - Auto-generates secure PostgreSQL password
# - Auto-generates secure JWT secret key
# - Sets up document directories
# - Creates optimized .env configuration
wizard:
	@python wizard_launcher.py

docker-up:
	@echo "Starting Semantik services with Docker Compose..."
	@echo "Setting up directories with correct permissions..."
	@mkdir -p ./models ./data ./logs
	@if command -v sudo >/dev/null 2>&1; then \
		sudo chown -R 1000:1000 ./models ./data ./logs; \
	else \
		chown -R 1000:1000 ./models ./data ./logs 2>/dev/null || echo "WARNING: Could not set directory permissions. If you encounter permission errors, run: sudo chown -R 1000:1000 ./models ./data ./logs"; \
	fi
	@echo "✓ Directories ready"
	@if [ ! -f .env ]; then \
		echo "Creating .env file from .env.docker.example..."; \
		cp .env.docker.example .env; \
		echo "Generating secure JWT_SECRET_KEY..."; \
		if command -v openssl >/dev/null 2>&1; then \
			JWT_KEY=$$(openssl rand -hex 32); \
			if [ "$$(uname)" = "Darwin" ]; then \
				sed -i '' "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$$JWT_KEY/" .env; \
			else \
				sed -i "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$$JWT_KEY/" .env; \
			fi; \
			echo "✓ Generated secure JWT_SECRET_KEY"; \
		else \
			echo "WARNING: openssl not found. Please manually set JWT_SECRET_KEY in .env"; \
		fi; \
	else \
		if grep -q "JWT_SECRET_KEY=CHANGE_THIS_TO_A_STRONG_SECRET_KEY" .env; then \
			echo "Detected default JWT_SECRET_KEY, generating secure key..."; \
			if command -v openssl >/dev/null 2>&1; then \
				JWT_KEY=$$(openssl rand -hex 32); \
				if [ "$$(uname)" = "Darwin" ]; then \
					sed -i '' "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$$JWT_KEY/" .env; \
				else \
					sed -i "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$$JWT_KEY/" .env; \
				fi; \
				echo "✓ Updated JWT_SECRET_KEY to secure value"; \
			else \
				echo "ERROR: Default JWT_SECRET_KEY detected but openssl not available"; \
				echo "Please manually set JWT_SECRET_KEY in .env"; \
				exit 1; \
			fi; \
		fi; \
	fi
	@if [ ! -f .env ] || ! grep -q "^POSTGRES_PASSWORD=" .env || [ "$$(grep '^POSTGRES_PASSWORD=' .env | cut -d'=' -f2)" = "" ]; then \
		echo "Generating secure POSTGRES_PASSWORD..."; \
		POSTGRES_PWD=$$(openssl rand -hex 32 2>/dev/null || echo "CHANGE_THIS_TO_A_STRONG_PASSWORD"); \
		if [ "$$(uname)" = "Darwin" ]; then \
			sed -i '' "s/POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$$POSTGRES_PWD/" .env; \
		else \
			sed -i "s/POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$$POSTGRES_PWD/" .env; \
		fi; \
		echo "✓ Generated secure POSTGRES_PASSWORD"; \
	fi
	docker compose up -d
	@echo "Services started! Access the application at http://localhost:8080"

docker-down:
	@echo "Stopping Semantik services..."
	docker compose down

docker-logs:
	docker compose logs -f

docker-build-fresh:
	@echo "Rebuilding Docker images without cache..."
	docker compose build --no-cache

docker-ps:
	docker compose ps

docker-restart:
	@echo "Restarting Semantik services..."
	docker compose restart

# Quick commands for individual services
docker-logs-webui:
	docker compose logs -f webui

docker-logs-vecpipe:
	docker compose logs -f vecpipe

docker-logs-qdrant:
	docker compose logs -f qdrant

# Shell access to containers
docker-shell-webui:
	docker compose exec webui /bin/bash

docker-shell-vecpipe:
	docker compose exec vecpipe /bin/bash

# PostgreSQL commands
docker-postgres-logs:
	docker compose logs -f postgres

docker-shell-postgres:
	docker compose exec postgres psql -U semantik -d semantik

docker-postgres-backup:
	@echo "Creating PostgreSQL backup..."
	@mkdir -p ./backups
	@BACKUP_FILE="./backups/semantik_backup_$$(date +%Y%m%d_%H%M%S).sql" && \
	docker compose exec -T postgres pg_dump -U semantik semantik > $$BACKUP_FILE && \
	echo "✓ Backup created: $$BACKUP_FILE"

docker-postgres-restore:
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "ERROR: Please specify BACKUP_FILE=path/to/backup.sql"; \
		exit 1; \
	fi
	@echo "Restoring PostgreSQL from $(BACKUP_FILE)..."
	@docker compose exec -T postgres psql -U semantik -d semantik < $(BACKUP_FILE)
	@echo "✓ Database restored from $(BACKUP_FILE)"

# Development shortcuts
.PHONY: fix check run

fix: format

check: lint type-check test

run:
	cd packages/webui && python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload

# Frontend commands
frontend-install:
	cd apps/webui-react && npm install

frontend-build:
	cd apps/webui-react && npm run build

frontend-dev:
	cd apps/webui-react && npm run dev

frontend-test:
	cd apps/webui-react && npm test -- --run

# Integrated commands
build: frontend-build

dev:
	./scripts/dev.sh

# Local development commands - Run webui locally with hot reload
dev-local:
	./scripts/dev-local.sh

docker-dev-up:
	@echo "Starting development services in Docker (backend only, for local webui development)..."
	@mkdir -p ./models ./data ./logs
	@if [ ! -f .env ]; then \
		echo "Creating .env file from .env.docker.example..."; \
		cp .env.docker.example .env; \
	fi
	docker compose --profile backend up -d
	@echo "Backend services started! Configure .env.local and run 'make run' to start webui locally"

docker-dev-down:
	@echo "Stopping development services..."
	docker compose --profile backend down

docker-dev-logs:
	docker compose --profile backend logs -f