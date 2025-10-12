FROM python:3.11-slim

WORKDIR /app

# Zainstaluj zależności systemowe
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Kopiuj requirements
COPY requirements.txt .

# Zainstaluj zależności Python
RUN pip install --no-cache-dir -r requirements.txt

# Kopiuj kod źródłowy
COPY src/ ./src/

# Expose port
EXPOSE 8002

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Uruchom aplikację
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002"]
