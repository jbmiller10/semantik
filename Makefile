# Makefile for Document Embedding System

.PHONY: help install dev-install format lint type-check test test-coverage clean
.PHONY: frontend-install frontend-build frontend-dev frontend-test build dev
.PHONY: docker-up docker-down docker-logs docker-build-fresh docker-ps docker-restart

help:
	@echo "Available commands:"
	@echo "  install        Install production dependencies"
	@echo "  dev-install    Install development dependencies"
	@echo "  format         Format code with black and isort"
	@echo "  lint           Run linting with ruff"
	@echo "  type-check     Run type checking with mypy"
	@echo "  test           Run tests"
	@echo "  test-coverage  Run tests with coverage report"
	@echo "  clean          Clean up generated files"
	@echo ""
	@echo "Docker commands:"
	@echo "  wizard            Interactive Docker setup wizard (TUI)"
	@echo "  docker-up         Start all services with docker-compose"
	@echo "  docker-down       Stop and remove all containers"
	@echo "  docker-logs       View logs from all services"
	@echo "  docker-build-fresh Rebuild images without cache"
	@echo "  docker-ps         Show status of all containers"
	@echo "  docker-restart    Restart all services"
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

install:
	poetry install --no-dev

dev-install:
	poetry install

format:
	black packages/vecpipe packages/webui tests
	isort packages/vecpipe packages/webui tests

lint:
	ruff check packages/vecpipe packages/webui tests

type-check:
	mypy packages/vecpipe packages/webui --ignore-missing-imports

test:
	pytest tests -v

test-coverage:
	pytest tests -v --cov=packages.vecpipe --cov=packages.webui --cov-report=html --cov-report=term

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov .pytest_cache .mypy_cache .ruff_cache
	rm -rf *.egg-info dist build

# Docker commands for the new setup
wizard:
	@python wizard_launcher.py

docker-up:
	@echo "Starting Semantik services with Docker Compose..."
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
	docker-compose logs -f webui

docker-logs-vecpipe:
	docker-compose logs -f vecpipe

docker-logs-qdrant:
	docker-compose logs -f qdrant

# Shell access to containers
docker-shell-webui:
	docker-compose exec webui /bin/bash

docker-shell-vecpipe:
	docker-compose exec vecpipe /bin/bash

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
	cd apps/webui-react && npm test

# Integrated commands
build: frontend-build

dev:
	./scripts/dev.sh