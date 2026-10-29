FROM python:3.10.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV MEMORY_DB_PATH=/app/artifacts/memory.db
ENV CODING_CONFIG_PATH=/app/configs/coding/default.json

COPY requirements-api.txt constraints-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt -c constraints-api.txt

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN mkdir -p /app/artifacts
VOLUME ["/app/artifacts"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10m --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["uvicorn", "insightmem.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
