.PHONY: help build run stop clean test logs

help: ## Pokaż pomoc
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Zbuduj obrazy Docker
	docker-compose build

run: ## Uruchom wszystkie serwisy
	docker-compose up -d

stop: ## Zatrzymaj wszystkie serwisy
	docker-compose down

clean: ## Wyczyść wszystko (UWAGA: usuwa dane)
	docker-compose down -v
	docker system prune -f

test: ## Uruchom testy
	python integrated-system.py --mode demo

logs: ## Pokaż logi
	docker-compose logs -f

status: ## Sprawdź status serwisów
	docker-compose ps

restart: ## Restartuj serwisy
	docker-compose restart

install: ## Zainstaluj zależności lokalnie
	pip install -r requirements.txt
