FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (Bada layer — pehle se cached hai)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Requirements copy
COPY requirements.txt .

# Network timeout aur retries ke sath dependencies install
RUN pip install --no-cache-dir --default-timeout=1000 --retries 10 -r requirements.txt

# Application code copy
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]