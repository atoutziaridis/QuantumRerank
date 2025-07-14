# Production-ready Dockerfile for QuantumRerank
# Simple deployment: docker run -p 8000:8000 quantumrerank/server:latest

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY quantum_rerank/ ./quantum_rerank/
COPY config/ ./config/

# Create non-root user
RUN useradd -m -u 1000 quantumrerank && \
    chown -R quantumrerank:quantumrerank /app

USER quantumrerank

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application with uvicorn directly (simpler than gunicorn for single container)
CMD ["python", "-m", "uvicorn", "quantum_rerank.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]