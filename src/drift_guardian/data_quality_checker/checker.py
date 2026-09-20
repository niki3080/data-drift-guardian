import logging
from datetime import datetime
from typing import Optional, Any

import pandas as pd
import numpy as np
from pydantic import BaseModel, ValidationError, create_model

logger = logging.getLogger("schema_checker")

class SchemaChecker:
    """
    Проверяет входящие события (dict из Kafka) или целые DataFrame'ы
    на соответствие схеме референсного pd.DataFrame.

    check_event  — per-event валидация через pydantic (значения + типы).
    check_df     — батчевая валидация DataFrame по dtypes колонок
                   (быстрее, но проверяет только типы, не значения).

    Ограничения:
      - nullable-int типы pandas (Int8, Int16, ..., Int64) не поддерживаются
        и приведут к ошибке при построении схемы — считаем, что в референсе
        их быть не должно.
      - datetime-колонки запрещены везде, кроме одной сконфигурированной
        колонки времени kafka-события (по умолчанию "event_time").
    """

    PANDAS_TO_PY = {
        "int8": int, "int16": int, "int32": int, "int64": int,
        "uint8": int, "uint16": int, "uint32": int, "uint64": int,
        "float16": float, "float32": float, "float64": float,
        "bool": bool,
        "object": str,
        "category": str,
    }

    UNSUPPORTED_DTYPES = {
        "Int8", "Int16", "Int32", "Int64",
        "UInt8", "UInt16", "UInt32", "UInt64",
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

    def check_event(self, event: dict[str, Any]) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
        """
        Валидирует одно событие.

        :return: (is_valid, validated_dict | None, error_message | None)
        """
        try:
            self._check_columns_coverage(set(event.keys()), source_desc="event")
            validated = self.schema_model(**event)
        except (ValidationError, ValueError) as e:
            logger.warning(f"Schema mismatch: {e}")
            return False, None, str(e)

        return True, validated.model_dump(), None

    def check_df(self, df: pd.DataFrame) -> None:
        """
        Валидирует целиком DataFrame (батч событий) на соответствие
        референсной схеме. В отличие от check_event, проверяет только
        dtypes колонок целиком, а не значения построчно через pydantic —
        значительно быстрее для больших DataFrame.

        Ничего не возвращает при успешной проверке (тихо проходит).

        :param df: DataFrame для проверки
        :raises ValueError: если отсутствует required-колонка либо её dtype
                             не совпадает с референсным (и raise_on_missing_required=True)
        """
        self._check_columns_coverage(set(df.columns), source_desc="DataFrame")
        self._check_dtypes_df(df)

    def _resolve_py_type(self, col: str, dtype: np.dtype) -> type:
        dtype_str = str(dtype)

        if dtype_str in self.UNSUPPORTED_DTYPES:
            raise ValueError(
                f"Column '{col}' has unsupported nullable dtype '{dtype_str}'. "
                f"Nullable int/bool types are not supported by SchemaChecker."
            )

        if pd.api.types.is_datetime64_any_dtype(dtype):
            if col != self.time_column:
                raise ValueError(
                    f"Column '{col}' has datetime dtype, but only the configured "
                    f"time_column='{self.time_column}' is allowed to be datetime."
                )
            return datetime

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
                fields[col] = (py_type, ...)
            else:
                fields[col] = (Optional[py_type], None)
        return create_model("EventSchema", **fields)

    def _check_columns_coverage(self, present_columns: set[str], source_desc: str = "event") -> None:
        missing = self.reference_columns - present_columns
        missing_required = missing & self.required_cols
        missing_optional = missing - self.required_cols

        if missing_required:
            msg = f"Missing REQUIRED columns in {source_desc}: {missing_required}"
            logger.error(msg)
            if self.raise_on_missing_required:
                raise ValueError(msg)

        if missing_optional:
            logger.warning(
                f"Missing optional columns in {source_desc} (excluded from drift calc): {missing_optional}"
            )

    def _check_dtypes_df(self, df: pd.DataFrame) -> None:
        for col, expected_dtype in self.reference_df.dtypes.items():
            if col not in df.columns:
                continue

            actual_dtype = df[col].dtype

            # Для category сравниваем только "форму" типа, а не конкретный
            # набор категорий/ordered — иначе разные наборы значений
            # ошибочно считаются несовместимыми типами.
            if isinstance(expected_dtype, pd.CategoricalDtype):
                dtypes_match = isinstance(actual_dtype, pd.CategoricalDtype)
            else:
                dtypes_match = pd.api.types.is_dtype_equal(actual_dtype, expected_dtype)

            if not dtypes_match:
                msg = (
                    f"Column '{col}' dtype mismatch: "
                    f"expected '{expected_dtype}', got '{actual_dtype}'"
                )
                if col in self.required_cols:
                    logger.error(msg)
                    if self.raise_on_missing_required:
                        raise ValueError(msg)
                else:
                    logger.warning(msg)





