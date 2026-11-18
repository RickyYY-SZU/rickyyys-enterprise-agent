FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements.txt .
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --index-url "$PIP_INDEX_URL" --upgrade pip \
    && pip install --index-url "$PIP_INDEX_URL" -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY static ./static
COPY data ./data

RUN mkdir -p /app/data && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["uvicorn", "enterprise_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]

