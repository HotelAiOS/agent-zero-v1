FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install basic requirements first
RUN pip install --no-cache-dir fastapi uvicorn pydantic httpx python-multipart

# Try to install advanced NLP libraries (with error handling)
RUN pip install --no-cache-dir spacy sentence-transformers scikit-learn numpy pandas || \
    echo "Advanced NLP libraries installation failed - using fallback mode"

# Try to download spaCy models (with error handling)
RUN python -m spacy download en_core_web_sm || \
    echo "spaCy model download failed - basic NLP mode will be used"

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/data /app/models

# Expose port
EXPOSE 8010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8010/health || exit 1

# Start application
CMD ["python", "main.py"]
