.PHONY: help dev-up dev-down build test lint

help:
	@echo "Agent Zero v1.0.0 - Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make dev-up      - Start local k3d cluster"
	@echo "  make dev-down    - Stop local k3d cluster"
	@echo "  make build       - Build all Docker images"
	@echo "  make test        - Run all tests"
	@echo "  make lint        - Run linters"
	@echo "  make deploy-dev  - Deploy to dev environment"

dev-up:
	k3d cluster create agent-zero-dev --agents 3 \
		--port 8080:80@loadbalancer \
		--port 8443:443@loadbalancer \
		--volume $(PWD)/data:/data

dev-down:
	k3d cluster delete agent-zero-dev

build:
	@echo "Building all services..."
	cd services/ai-router && docker build -t agent-zero/ai-router:1.0.0 .
	cd services/chat-service && docker build -t agent-zero/chat-service:1.0.0 .
	cd services/agent-orchestrator && docker build -t agent-zero/agent-orchestrator:1.0.0 .

test:
	pytest tests/ -v --cov=services

lint:
	black services/
	flake8 services/
	mypy services/

deploy-dev:
	kubectl apply -k infrastructure/kubernetes/overlays/dev
