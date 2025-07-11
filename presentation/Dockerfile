FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY presentation/ presentation/
COPY use_cases/      use_cases/
COPY repositories/    repositories/
COPY entities/        entities/
COPY config.py        config.py

# Expose the FastAPI port
EXPOSE 8000

# Default command to run the API
CMD ["uvicorn", "presentation.api:app", "--host", "0.0.0.0", "--port", "8000"]