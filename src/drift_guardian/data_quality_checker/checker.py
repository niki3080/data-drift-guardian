import logging
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError, create_model

logger = logging.getLogger("schema_checker")


class SchemaChecker:
    """
    Проверяет входящие события (dict из Kafka) на соответствие схеме
    референсного pd.DataFrame. Работает per-event, без буферизации
    и без построения DataFrame на каждое сообщение.

    Ограничения:
      - nullable-int типы pandas (Int8, Int16, ..., Int64) не поддерживаются
        и приведут к ошибке при построении схемы — считаем, что в референсе
        их быть не должно.
      - datetime-колонки запрещены везде, кроме одной сконфигурированной
        колонки времени kafka-события (по умолчанию "event_time").
    """

    # даункаст-типы numpy/pandas -> python-типы для pydantic
    PANDAS_TO_PY = {
        # signed int
        "int8": int,
        "int16": int,
        "int32": int,
        "int64": int,
        # unsigned int
        "uint8": int,
        "uint16": int,
        "uint32": int,
        "uint64": int,
        # float
        "float16": float,
        "float32": float,
        "float64": float,
        # прочее
        "bool": bool,
        "object": str,
        "category": str,
    }

    # nullable pandas dtypes, которые явно не поддерживаем
    UNSUPPORTED_DTYPES = {
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "boolean",  # nullable bool
    }

    def __init__(
        self,
        reference_df: pd.DataFrame,
        required_cols: Optional[set[str]] = None,
        time_column: str = "event_time",
        raise_on_missing_required: bool = True,
    ):
        """
        :param reference_df: референсный датафрейм, из dtypes которого строится схема
        :param required_cols: колонки, критичные для инференса (AV) — при их
                               отсутствии/несовпадении типа будет Error
        :param time_column: единственная разрешённая временная колонка
                             (обычно время kafka-события)
        :param raise_on_missing_required: кидать исключение (True) или
                               только логировать критическую ошибку (False)
        """
        self.reference_df = reference_df
        self.required_cols = required_cols or set()
        self.time_column = time_column
        self.raise_on_missing_required = raise_on_missing_required

        self.reference_columns = set(reference_df.columns)
        self.schema_model: type[BaseModel] = self._build_schema_model()

    # ------------------------------------------------------------------ #
    # Построение схемы
    # ------------------------------------------------------------------ #
    def _resolve_py_type(self, col: str, dtype: np.dtype) -> type:
        dtype_str = str(dtype)

        # nullable int/bool — явно запрещаем, чтобы не приводить молча
        if dtype_str in self.UNSUPPORTED_DTYPES:
            raise ValueError(
                f"Column '{col}' has unsupported nullable dtype '{dtype_str}'. "
                f"Nullable int/bool types are not supported by SchemaChecker."
            )

        # datetime — разрешён только для сконфигурированной колонки времени
        if pd.api.types.is_datetime64_any_dtype(dtype):
            if col != self.time_column:
                raise ValueError(
                    f"Column '{col}' has datetime dtype, but only the configured "
                    f"time_column='{self.time_column}' is allowed to be datetime."
                )
            return datetime

        # строковые dtype pandas приводим к обычному Python str
        if pd.api.types.is_string_dtype(dtype):
            return str

        py_type = self.PANDAS_TO_PY.get(dtype_str)
        if py_type is None:
            raise ValueError(
                f"Column '{col}' has unmapped dtype '{dtype_str}'. "
                f"Add it explicitly to PANDAS_TO_PY."
            )
        return py_type

    def _build_schema_model(self) -> type[BaseModel]:
        fields = {}
        for col, dtype in self.reference_df.dtypes.items():
            py_type = self._resolve_py_type(col, dtype)
            if col in self.required_cols:
                fields[col] = (py_type, ...)  # обязательное поле
            else:
                fields[col] = (Optional[py_type], None)  # опциональное
        return create_model("EventSchema", **fields)

    def _check_columns_coverage(self, event: dict[str, Any]) -> None:
        event_columns = set(event.keys())
        missing = self.reference_columns - event_columns
        missing_required = missing & self.required_cols
        missing_optional = missing - self.required_cols

        if missing_required:
            msg = f"Missing REQUIRED columns in event: {missing_required}"
            if self.raise_on_missing_required:
                raise ValueError(msg)
            logger.error(msg)

        if missing_optional:
            logger.warning(
                "Missing optional columns (excluded from drift calc): %s",
                missing_optional,
            )

    def check_event(
        self,
        event: dict[str, Any],
    ) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
        """
        Валидирует одно событие.

        :return: (is_valid, validated_dict | None, error_message | None)
        """
        try:
            self._check_columns_coverage(event)
            validated = self.schema_model(**event)
        except (ValidationError, ValueError) as e:
            logger.warning("Schema mismatch: %s", e)
            return False, None, str(e)

        return True, validated.model_dump(), None
