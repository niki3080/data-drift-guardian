from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any


class InvalidEventTime(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class KafkaEvent:
    event_id: str | int
    event_time: datetime
    features: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> KafkaEvent:
        event_id = payload.get("event_id")
        if isinstance(event_id, bool) or not isinstance(event_id, (str, int)):
            raise ValueError("event_id must be a string or integer")

        event_time = cls._parse_event_time(payload.get("event_time"))
        features = {
            name: value
            for name, value in payload.items()
            if name not in {"event_id", "event_time"}
        }

        if not features:
            raise ValueError("event must contain at least one feature")

        for name, value in features.items():
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError(f"feature {name!r} must be a JSON scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"feature {name!r} must be finite")

        prediction_score = features.get("prediction_score")
        if prediction_score is not None:
            if isinstance(prediction_score, bool) or not isinstance(
                prediction_score,
                (int, float),
            ):
                raise ValueError("prediction_score must be numeric")
            if not 0 <= float(prediction_score) <= 1:
                raise ValueError("prediction_score must be in [0, 1]")

        return cls(
            event_id=event_id,
            event_time=event_time,
            features=features,
        )

    @staticmethod
    def _parse_event_time(value: Any) -> datetime:
        if not isinstance(value, str):
            raise InvalidEventTime("event_time must be an ISO-8601 string")

        try:
            event_time = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise InvalidEventTime("event_time must be valid ISO-8601") from exc

        if event_time.tzinfo is None or event_time.utcoffset() is None:
            raise InvalidEventTime("event_time must include a timezone")

        return event_time
