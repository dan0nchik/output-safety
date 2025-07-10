FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY workers/aggregator/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY workers/aggregator/aggregator.py workers/aggregator/aggregator.py
COPY repositories/            repositories/
COPY use_cases/               use_cases/
COPY entities/                entities/
COPY config.py                config.py

# Set PYTHONPATH so shared modules are visible
ENV PYTHONPATH=/app

# Run aggregator worker
CMD ["python", "-u", "workers/aggregator/aggregator.py"]
