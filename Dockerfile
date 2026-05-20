FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir --default-timeout=120 --retries=10 -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["python", "run_pipeline.py"]