FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY oanda_client.py .
COPY pairs_analyzer.py .
COPY main.py .

# Run as non-root user (required for OpenShift)
RUN useradd -m -u 1001 botuser
USER 1001

# Default command
CMD ["python", "-u", "main.py"]
