from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from domain.enums import DatasetColumnRole


@dataclass(frozen=True)
class RegressionColumn:
    key: str
    label: str
    role: DatasetColumnRole = DatasetColumnRole.PREDICTOR

    def to_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "label": self.label,
            "role": self.role.value,
        }


@dataclass(frozen=True)
class RegressionRow:
    cells: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {"cells": list(self.cells)}


@dataclass(frozen=True)
class RegressionDataset:
    columns: tuple[RegressionColumn, ...]
    rows: tuple[RegressionRow, ...]
    source_mode: str = "grid"
    filename: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "columns": [column.to_dict() for column in self.columns],
            "rows": [row.to_dict() for row in self.rows],
            "sourceMode": self.source_mode,
            "filename": self.filename,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RegressionDataset:
        return cls(
            columns=tuple(
                RegressionColumn(
                    key=str(column.get("key") or f"column_{index + 1}"),
                    label=str(column.get("label") or "").strip(),
                    role=DatasetColumnRole(str(column.get("role") or DatasetColumnRole.PREDICTOR.value)),
                )
                for index, column in enumerate(payload.get("columns", []))
            ),
            rows=tuple(
                RegressionRow(cells=tuple("" if cell is None else str(cell).strip() for cell in row.get("cells", [])))
                for row in payload.get("rows", [])
            ),
            source_mode=str(payload.get("sourceMode") or "grid"),
            filename=str(payload.get("filename") or ""),
        )

    def with_filled_cells(self, replacements: dict[tuple[int, int], str]) -> RegressionDataset:
        updated_rows = []
        for row_index, row in enumerate(self.rows):
            updated_cells = []
            for column_index, cell in enumerate(row.cells):
                updated_cells.append(replacements.get((row_index, column_index), cell))
            updated_rows.append(RegressionRow(cells=tuple(updated_cells)))
        return replace(self, rows=tuple(updated_rows))


@dataclass(frozen=True)
class PredictionRowResult:
    row_number: int
    values: tuple[str, ...]

    def to_table_row(self) -> tuple[str, ...]:
        return (str(self.row_number), *self.values)


@dataclass(frozen=True)
class MatchingPairResult:
    treated_id: str
    control_id: str
    treated_score: float
    control_score: float
    distance: float
    treated_outcome: float
    control_outcome: float

    def to_table_row(self) -> tuple[str, ...]:
        return (
            self.treated_id,
            self.control_id,
            f"{self.treated_score:.6g}",
            f"{self.control_score:.6g}",
            f"{self.distance:.6g}",
            f"{self.treated_outcome:.6g}",
            f"{self.control_outcome:.6g}",
            f"{(self.treated_outcome - self.control_outcome):.6g}",
        )


@dataclass(frozen=True)
class PreparedRegressionDataset:
    dataset: RegressionDataset
    predictor_columns: tuple[RegressionColumn, ...]
    target_column: RegressionColumn
    id_column: RegressionColumn | None
    training_frame: pd.DataFrame
    prediction_frame: pd.DataFrame
    training_row_indices: tuple[int, ...]
    prediction_row_indices: tuple[int, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PreparedMatchingDataset:
    dataset: RegressionDataset
    predictor_columns: tuple[RegressionColumn, ...]
    treatment_column: RegressionColumn
    outcome_column: RegressionColumn
    id_column: RegressionColumn | None
    dataframe: pd.DataFrame
    warnings: tuple[str, ...] = field(default_factory=tuple)
