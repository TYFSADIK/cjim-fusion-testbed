# The pipeline workers and the dashboard API. vLLM is not in this image: it runs on the host's GPU.
ARG BASE=python:3.12-slim
FROM ${BASE}
RUN pip install --no-cache-dir confluent-kafka==2.6.1 "psycopg[binary]==3.2.3" httpx==0.28.1 fastapi==0.115.5 uvicorn==0.32.1
WORKDIR /app
COPY fusion /app/fusion
COPY api /app/api
COPY scenario /app/scenario
COPY eval /app/eval
ENV PYTHONUNBUFFERED=1
