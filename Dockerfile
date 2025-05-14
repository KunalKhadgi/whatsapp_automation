# Use slim Python base image
FROM python:3.10-slim

# Set essential environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080

# Create app user (non-root)
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to use Docker layer cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Clean up pyc files and create required dirs
RUN find . -name '*.pyc' -delete && \
    mkdir -p /app/exported_data /app/processed_exported_data /app/faiss_db /app/processed_zips && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set production environment
ENV FLASK_ENV=production \
    FLASK_APP=unified_app.py

# Healthcheck endpoint (OPTIONAL: for faster diagnosis)
HEALTHCHECK CMD curl -f http://localhost:8080/ || exit 1

# Run the app with Gunicorn (Cloud Run expects it to bind to $PORT)
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "8", "--timeout", "300", "unified_app:app"]
