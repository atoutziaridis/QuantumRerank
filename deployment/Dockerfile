# Multi-stage Docker build for QuantumRerank
# Optimized for production deployment with security hardening

# Stage 1: Base image with system dependencies
FROM python:3.11-slim AS base

# Set build arguments
ARG ENVIRONMENT=production
ARG VERSION=1.0.0

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    ENVIRONMENT=${ENVIRONMENT} \
    VERSION=${VERSION}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r quantum && useradd -r -g quantum -m -s /bin/bash quantum

# Stage 2: Dependencies installation
FROM base AS dependencies

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt requirements-prod.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -r requirements-prod.txt && \
    pip install uvicorn[standard] gunicorn

# Stage 3: Production image
FROM base AS production

# Copy virtual environment from dependencies stage
COPY --from=dependencies /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY quantum_rerank/ ./quantum_rerank/
COPY config/ ./config/
COPY deployment/environments/ ./environments/

# Copy startup scripts
COPY scripts/start.sh ./scripts/
COPY scripts/health-check.sh ./scripts/

# Make scripts executable
RUN chmod +x ./scripts/*.sh

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/data /app/cache && \
    chown -R quantum:quantum /app && \
    chmod -R 755 /app

# Switch to non-root user
USER quantum

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ./scripts/health-check.sh || exit 1

# Default command
CMD ["./scripts/start.sh"]

# Stage 4: Development image (optional)
FROM dependencies AS development

# Install development dependencies
COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/logs /app/data /app/cache

# Set development environment
ENV ENVIRONMENT=development \
    DEBUG=true

# Development user setup
USER quantum
WORKDIR /app

# Development command with hot reload
CMD ["uvicorn", "quantum_rerank.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "quantum_rerank"]