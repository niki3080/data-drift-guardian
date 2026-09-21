FROM ghcr.io/astral-sh/uv:0.10.0 AS uv
FROM python:3.13-slim

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# LightGBM links against the GNU OpenMP runtime on Linux.
# python:* -slim images do not include libgomp by default.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["python", "-m", "drift_guardian.ingestion.consumer"]
