# Makefile for Document Embedding System

.PHONY: help install dev-install format lint type-check test test-coverage clean docker-build docker-run
.PHONY: frontend-install frontend-build frontend-dev frontend-test build dev

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
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run Docker container"
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

docker-build:
	docker build -f Dockerfile.webui -t document-embedding-webui:latest .

docker-run:
	docker run -d \
		--name embedding-webui \
		-p 8080:8080 \
		-e QDRANT_HOST=192.168.1.173 \
		-e QDRANT_PORT=6333 \
		-v /mnt/docs:/mnt/docs:ro \
		-v /var/embeddings:/var/embeddings \
		document-embedding-webui:latest

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