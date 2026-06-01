FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

COPY app/    ./app/
COPY data/   ./data/
COPY start.sh ./start.sh

RUN mkdir -p /app/vector_store /app/logs && chmod +x /app/start.sh

RUN useradd -m -u 1001 appuser \
    && chown -R appuser /app \
    && mkdir -p /home/appuser/.cache \
    && cp -r /root/.cache/huggingface /home/appuser/.cache/huggingface \
    && chown -R appuser /home/appuser/.cache

USER appuser

ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/home/appuser/.cache/huggingface

EXPOSE 8000

CMD ["/bin/bash", "/app/start.sh"]