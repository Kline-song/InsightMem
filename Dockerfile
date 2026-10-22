FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV MEMORY_DB_PATH=/app/artifacts/memory.db
ENV MEMORY_RETRIEVAL_MODE=enhanced
ENV MEMORY_RRF_K=60
ENV MEMORY_LEXICAL_WEIGHT=0.25
ENV MEMORY_CONTEXT_MESSAGES=1
ENV MEMORY_OPTION_VIEW_WEIGHT=0.2

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data
COPY scripts ./scripts

RUN mkdir -p /app/artifacts
VOLUME ["/app/artifacts"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10m --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["uvicorn", "insightmem.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
