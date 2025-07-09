FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy code for worker and shared modules
COPY workers/safety/safety_worker.py   workers/safety/safety_worker.py
COPY repositories/            repositories/
COPY use_cases/               use_cases/
COPY entities/                entities/
COPY config.py                config.py
ENV PYTHONPATH=/app
# No ports exposed — this is a background worker
# Run the PII worker script
CMD ["python", "-u", "workers/safety/safety_worker.py"]