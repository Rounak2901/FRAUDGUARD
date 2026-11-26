FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1
ENV FRAUD_GUARD_PORT=8001

# Demo-ready artifacts (CPU). Full academic train: docker compose run api python -m src.train
RUN python -m src.data.download --synthetic \
 && python -m src.data.preprocess \
 && python -m src.train_web_bundle

EXPOSE 8001
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
