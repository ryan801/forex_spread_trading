FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY oanda_client.py .
COPY pairs_analyzer.py .
COPY main.py .

RUN useradd -m -u 1001 botuser
USER 1001

CMD ["python", "-u", "main.py"]