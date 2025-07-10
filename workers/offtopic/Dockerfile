FROM python:3.10-slim

# Set working directory
WORKDIR /app

# # Install system dependencies
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY workers/offtopic/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Copy code for worker and shared modules
COPY workers/offtopic/off_topic_worker.py   workers/offtopic/off_topic_worker.py
COPY repositories/            repositories/
COPY use_cases/               use_cases/
COPY entities/                entities/
COPY config.py                config.py
ENV PYTHONPATH=/app
# No ports exposed — this is a background worker
# Run the PII worker script
CMD ["python", "-u", "workers/offtopic/off_topic_worker.py"]