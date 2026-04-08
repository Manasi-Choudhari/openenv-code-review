# ── Build stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Metadata
LABEL org.opencontainers.image.title="GitHub PR Code Review Environment"
LABEL org.opencontainers.image.description="OpenEnv environment for PR code review"
LABEL org.opencontainers.image.version="1.0.0"

# Environment hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Application ────────────────────────────────────────────────────────────
WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/        ./app/
COPY server.py   ./server.py
COPY inference.py ./inference.py
COPY openenv.yaml ./openenv.yaml

# ── Runtime ────────────────────────────────────────────────────────────────
# HuggingFace Spaces expects port 7860
EXPOSE 7860

# Healthcheck — verifies the server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Default: start the API server
# To run inference: docker run --env-file .env <image> python inference.py
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
