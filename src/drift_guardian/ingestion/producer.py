import json
import os
import random
import signal
from datetime import UTC, datetime
from threading import Event

from confluent_kafka import Producer

STOP = Event()


def stop(*_: object) -> None:
    STOP.set()


def build_event(event_id: int, rng: random.Random) -> dict[str, object]:
    return {
        "event_id": event_id,
        "event_time": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "age": rng.randint(18, 75),
        "income": round(rng.uniform(30_000, 150_000), 2),
        "country": rng.choice(["DE", "FR", "EE"]),
        "prediction_score": round(rng.random(), 6),
    }


def main() -> None:
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:19092")
    topic = os.getenv("KAFKA_TOPIC", "features-stream")
    interval = float(os.getenv("PRODUCER_INTERVAL_SECONDS", "0.2"))

    producer = Producer({"bootstrap.servers": bootstrap_servers})
    rng = random.Random(42)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    event_id = 1
    try:
        while not STOP.is_set():
            event = build_event(event_id, rng)
            producer.produce(
                topic,
                key=str(event_id),
                value=json.dumps(event).encode(),
            )
            producer.poll(0)
            event_id += 1
            STOP.wait(interval)
    finally:
        producer.flush(10)


if __name__ == "__main__":
    main()
