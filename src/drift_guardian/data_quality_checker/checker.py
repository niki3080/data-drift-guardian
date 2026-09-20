import logging
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError, create_model

logger = logging.getLogger("schema_checker")


class SchemaChecker:
    """
    Проверяет входящие события (dict из Kafka) или целые DataFrame'ы
    на соответствие схеме референсного pd.DataFrame.

    check_event — per-event валидация через pydantic (значения + типы).
    check_df — батчевая валидация DataFrame по dtypes колонок
    (быстрее, но проверяет только типы, не значения).

    Ограничения:
      - nullable-int типы pandas (Int8, Int16, ..., Int64) не поддерживаются
        и приведут к ошибке при построении схемы — считаем, что в референсе
        их быть не должно;
      - datetime-колонки запрещены везде, кроме одной сконфигурированной
        колонки времени Kafka-события (по умолчанию "event_time").
    """

    PANDAS_TO_PY = {
        "int8": int,
        "int16": int,
        "int32": int,
        "int64": int,
        "uint8": int,
        "uint16": int,
        "uint32": int,
        "uint64": int,
        "float16": float,
        "float32": float,
        "float64": float,
        "bool": bool,
        "object": str,
        "category": str,
    }

    UNSUPPORTED_DTYPES = {
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "boolean",
    }

    def __init__(
        self,
        reference_df: pd.DataFrame,
        required_cols: Optional[set[str]] = None,
        time_column: str = "event_time",
        raise_on_missing_required: bool = True,
    ):
        self.reference_df = reference_df
        self.required_cols = required_cols or set()
        self.time_column = time_column
        self.raise_on_missing_required = raise_on_missing_required

        self.reference_columns = set(reference_df.columns)
        self.schema_model: type[BaseModel] = self._build_schema_model()

    def check_event(
        self,
        event: dict[str, Any],
    ) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
        """
        Валидирует одно событие.

        :return: (is_valid, validated_dict | None, error_message | None)
        """
        try:
            self._check_columns_coverage(set(event), source_desc="event")
            validated = self.schema_model(**event)
        except (ValidationError, ValueError) as exc:
            logger.warning("Schema mismatch: %s", exc)
            return False, None, str(exc)

        return True, validated.model_dump(), None

    def check_df(self, df: pd.DataFrame) -> None:
        """Проверяет DataFrame на соответствие референсной схеме.

        Проверка выполняется по колонкам и dtype без построчной валидации
        значений через pydantic. Required-колонки приводят к ValueError,
        optional-колонки только логируют предупреждение.
        """
        self._check_columns_coverage(set(df.columns), source_desc="DataFrame")
        self._check_dtypes_df(df)

    def _resolve_py_type(self, col: str, dtype: np.dtype) -> type:
        dtype_str = str(dtype)

        if dtype_str in self.UNSUPPORTED_DTYPES:
            raise ValueError(
                f"Column '{col}' has unsupported nullable dtype '{dtype_str}'. "
                "Nullable int/bool types are not supported by SchemaChecker."
            )

        if pd.api.types.is_datetime64_any_dtype(dtype):
            if col != self.time_column:
                raise ValueError(
                    f"Column '{col}' has datetime dtype, but only the configured "
                    f"time_column='{self.time_column}' is allowed to be datetime."
                )
            return datetime

        # pandas StringDtype сериализуется как "string" и отсутствует
        # в PANDAS_TO_PY, поэтому строковые dtype обрабатываем отдельно.
        if pd.api.types.is_string_dtype(dtype):
            return str

        py_type = self.PANDAS_TO_PY.get(dtype_str)
        if py_type is None:
            raise ValueError(
                f"Column '{col}' has unmapped dtype '{dtype_str}'. "
                "Add it explicitly to PANDAS_TO_PY."
            )
        return py_type

    def _build_schema_model(self) -> type[BaseModel]:
        fields = {}
        for col, dtype in self.reference_df.dtypes.items():
            py_type = self._resolve_py_type(col, dtype)
            if col in self.required_cols:
                fields[col] = (py_type, ...)
            else:
                fields[col] = (Optional[py_type], None)
        return create_model("EventSchema", **fields)

    def _check_columns_coverage(
        self,
        present_columns: set[str],
        source_desc: str = "event",
    ) -> None:
        missing = self.reference_columns - present_columns
        missing_required = missing & self.required_cols
        missing_optional = missing - self.required_cols

        if missing_required:
            message = f"Missing REQUIRED columns in {source_desc}: {missing_required}"
            logger.error(message)
            if self.raise_on_missing_required:
                raise ValueError(message)

        if missing_optional:
            logger.warning(
                "Missing optional columns in %s (excluded from drift calc): %s",
                source_desc,
                missing_optional,
            )

    def _check_dtypes_df(self, df: pd.DataFrame) -> None:
        for col, expected_dtype in self.reference_df.dtypes.items():
            if col not in df.columns:
                continue

            actual_dtype = df[col].dtype

            # Для category сравниваем тип колонки, а не конкретный набор
            # категорий/ordered: разные значения категорий не меняют тип фичи.
            if isinstance(expected_dtype, pd.CategoricalDtype):
                dtypes_match = isinstance(actual_dtype, pd.CategoricalDtype)
            else:
                dtypes_match = pd.api.types.is_dtype_equal(
                    actual_dtype,
                    expected_dtype,
                )

            if dtypes_match:
                continue

            message = (
                f"Column '{col}' dtype mismatch: "
                f"expected '{expected_dtype}', got '{actual_dtype}'"
            )
            if col in self.required_cols:
                logger.error(message)
                if self.raise_on_missing_required:
                    raise ValueError(message)
            else:
                logger.warning(message)
