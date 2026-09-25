FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    confluent-kafka==2.6.1 \
    pandas==2.2.3 \
    pyyaml==6.0.2

COPY producer.py drift_config.yaml ./

CMD ["python", "producer.py"]
