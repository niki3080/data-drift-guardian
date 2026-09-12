FROM ghcr.io/astral-sh/uv:0.10.0 AS uv
FROM python:3.13-slim

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY drift_guardian ./drift_guardian

EXPOSE 8000

CMD ["python", "-m", "drift_guardian.realtime.consumer"]
