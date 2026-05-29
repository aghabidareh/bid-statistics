from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

import pandas as pd

from domain.enums import DatasetColumnRole
from domain.regression_inputs import PreparedMatchingDataset, PreparedRegressionDataset, RegressionColumn, RegressionDataset, RegressionRow
from domain.results import ValidationIssue
from services.validators import build_error, raise_if_issues

_ALLOWED_ROLES = {role.value for role in DatasetColumnRole}
TARGET_KIND_NUMERIC = "numeric"
TARGET_KIND_CATEGORICAL = "categorical"



def parse_regression_dataset(payload: object, field: str = "dataset") -> tuple[RegressionDataset | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    raw_payload: Any = payload
    if isinstance(payload, str):
        if not payload.strip():
            return None, [build_error(field, "Dataset payload is required.")]
        try:
            raw_payload = json.loads(payload)
        except json.JSONDecodeError:
            return None, [build_error(field, "Dataset payload must be valid JSON.")]

    if not isinstance(raw_payload, Mapping):
        return None, [build_error(field, "Dataset payload must be an object.")]

    columns_payload = raw_payload.get("columns")
    rows_payload = raw_payload.get("rows")
    if not isinstance(columns_payload, list) or not columns_payload:
        issues.append(build_error(f"{field}.columns", "Add at least one dataset column."))
        columns_payload = []
    if not isinstance(rows_payload, list) or not rows_payload:
        issues.append(build_error(f"{field}.rows", "Add at least one dataset row."))
        rows_payload = []

    columns: list[RegressionColumn] = []
    seen_labels: set[str] = set()

    for index, column_payload in enumerate(columns_payload):
        if not isinstance(column_payload, Mapping):
            issues.append(build_error(f"{field}.columns.{index}", "Each column must be an object."))
            continue

        label = str(column_payload.get("label") or "").strip()
        role_value = str(column_payload.get("role") or DatasetColumnRole.PREDICTOR.value)
        key = str(column_payload.get("key") or f"column_{index + 1}")

        if not label:
            issues.append(build_error(f"{field}.columns.{index}.label", "Column name is required."))
        normalized_label = label.casefold()
        if label and normalized_label in seen_labels:
            issues.append(build_error(f"{field}.columns.{index}.label", f"Column name '{label}' is duplicated."))
        elif label:
            seen_labels.add(normalized_label)

        if role_value not in _ALLOWED_ROLES:
            issues.append(build_error(f"{field}.columns.{index}.role", "Choose a valid column role."))
            role = DatasetColumnRole.PREDICTOR
        else:
            role = DatasetColumnRole(role_value)

        columns.append(RegressionColumn(key=key, label=label, role=role))

    rows: list[RegressionRow] = []
    expected_width = len(columns)
    for row_index, row_payload in enumerate(rows_payload):
        if not isinstance(row_payload, Mapping):
            issues.append(build_error(f"{field}.rows.{row_index}", "Each row must be an object."))
            continue
        cells_payload = row_payload.get("cells")
        if not isinstance(cells_payload, list):
            issues.append(build_error(f"{field}.rows.{row_index}.cells", "Each row must include a cell list."))
            continue
        if expected_width and len(cells_payload) != expected_width:
            issues.append(
                build_error(
                    f"{field}.rows.{row_index}.cells",
                    f"Row {row_index + 1} must contain exactly {expected_width} cells.",
                )
            )
            continue
        rows.append(RegressionRow(cells=tuple("" if cell is None else str(cell).strip() for cell in cells_payload)))

    return (
        RegressionDataset(
            columns=tuple(columns),
            rows=tuple(rows),
            source_mode=str(raw_payload.get("sourceMode") or "grid"),
            filename=str(raw_payload.get("filename") or ""),
        )
        if not issues
        else None,
        issues,
    )



def validate_role_counts(
    dataset: RegressionDataset,
    *,
    field: str = "dataset",
    required_roles: Mapping[DatasetColumnRole, int],
    maximum_roles: Mapping[DatasetColumnRole, int] | None = None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    role_counts = Counter(column.role for column in dataset.columns)

    for role, expected_count in required_roles.items():
        actual = role_counts.get(role, 0)
        if actual < expected_count:
            label = role.value.replace("_", " ")
            issues.append(build_error(f"{field}.columns", f"Select {expected_count} column(s) with the '{label}' role."))

    if maximum_roles is not None:
        for role, maximum_count in maximum_roles.items():
            actual = role_counts.get(role, 0)
            if actual > maximum_count:
                label = role.value.replace("_", " ")
                issues.append(build_error(f"{field}.columns", f"Use at most {maximum_count} column(s) with the '{label}' role."))

    return issues



def _is_empty_row(row: RegressionRow) -> bool:
    return all(not cell.strip() for cell in row.cells)



def _row_identifier(row: RegressionRow, row_index: int, id_index: int | None) -> str:
    if id_index is None:
        return str(row_index + 1)
    value = row.cells[id_index].strip()
    return value if value else str(row_index + 1)



def _parse_numeric_predictors(
    *,
    row: RegressionRow,
    row_index: int,
    field: str,
    predictor_columns: tuple[RegressionColumn, ...],
    predictor_indices: list[int],
    issues: list[ValidationIssue],
) -> dict[str, float] | None:
    predictor_values: dict[str, float] = {}
    has_errors = False
    for predictor_column, predictor_index in zip(predictor_columns, predictor_indices, strict=True):
        cell_value = row.cells[predictor_index].strip()
        if not cell_value:
            issues.append(build_error(f"{field}.rows.{row_index}.cells.{predictor_index}", f"{predictor_column.label} is required on row {row_index + 1}."))
            has_errors = True
            continue
        try:
            predictor_values[predictor_column.key] = float(cell_value)
        except ValueError:
            issues.append(build_error(f"{field}.rows.{row_index}.cells.{predictor_index}", f"{predictor_column.label} must be numeric on row {row_index + 1}."))
            has_errors = True
    return None if has_errors else predictor_values



def prepare_supervised_dataset(
    dataset: RegressionDataset,
    *,
    field: str = "dataset",
    min_training_rows: int = 3,
    min_prediction_rows: int = 0,
    min_predictor_columns: int = 1,
    max_predictor_columns: int | None = None,
    target_kind: str = TARGET_KIND_NUMERIC,
    minimum_classes: int | None = None,
    maximum_classes: int | None = None,
) -> PreparedRegressionDataset:
    issues = validate_role_counts(
        dataset,
        field=field,
        required_roles={DatasetColumnRole.TARGET: 1, DatasetColumnRole.PREDICTOR: min_predictor_columns},
        maximum_roles={DatasetColumnRole.TARGET: 1} | ({DatasetColumnRole.PREDICTOR: max_predictor_columns} if max_predictor_columns is not None else {}),
    )
    if issues:
        raise_if_issues(issues)

    predictor_columns = tuple(column for column in dataset.columns if column.role is DatasetColumnRole.PREDICTOR)
    target_column = next(column for column in dataset.columns if column.role is DatasetColumnRole.TARGET)
    id_column = next((column for column in dataset.columns if column.role is DatasetColumnRole.ID), None)

    training_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    training_row_indices: list[int] = []
    prediction_row_indices: list[int] = []
    warnings: list[str] = []

    target_index = dataset.columns.index(target_column)
    predictor_indices = [dataset.columns.index(column) for column in predictor_columns]
    id_index = dataset.columns.index(id_column) if id_column is not None else None

    for row_index, row in enumerate(dataset.rows):
        if len(row.cells) != len(dataset.columns):
            issues.append(build_error(f"{field}.rows.{row_index}.cells", f"Row {row_index + 1} has the wrong number of cells."))
            continue

        if _is_empty_row(row):
            warnings.append(f"Ignored empty row {row_index + 1}.")
            continue

        predictor_values = _parse_numeric_predictors(
            row=row,
            row_index=row_index,
            field=field,
            predictor_columns=predictor_columns,
            predictor_indices=predictor_indices,
            issues=issues,
        )
        if predictor_values is None:
            continue

        target_value = row.cells[target_index].strip()
        base_row = {
            **predictor_values,
            "__row_index__": row_index,
            "__row_id__": _row_identifier(row, row_index, id_index),
        }
        if not target_value:
            prediction_rows.append(base_row)
            prediction_row_indices.append(row_index)
            continue

        if target_kind == TARGET_KIND_NUMERIC:
            try:
                parsed_target: Any = float(target_value)
            except ValueError:
                issues.append(build_error(f"{field}.rows.{row_index}.cells.{target_index}", f"{target_column.label} must be numeric on row {row_index + 1}."))
                continue
        else:
            parsed_target = target_value

        training_rows.append({**base_row, target_column.key: parsed_target})
        training_row_indices.append(row_index)

    if len(training_rows) < min_training_rows:
        issues.append(build_error(f"{field}.rows", f"Provide at least {min_training_rows} training row(s) with a filled target value."))
    if len(prediction_rows) < min_prediction_rows:
        issues.append(build_error(f"{field}.rows", f"Provide at least {min_prediction_rows} prediction row(s) with a blank target value."))

    training_frame = pd.DataFrame(training_rows)
    prediction_frame = pd.DataFrame(prediction_rows)

    if minimum_classes is not None and not training_frame.empty:
        class_count = int(training_frame[target_column.key].nunique())
        if class_count < minimum_classes:
            issues.append(build_error(f"{field}.rows", f"{target_column.label} must contain at least {minimum_classes} classes in the training rows."))
        if maximum_classes is not None and class_count > maximum_classes:
            issues.append(build_error(f"{field}.rows", f"{target_column.label} must contain no more than {maximum_classes} classes in the training rows."))

    raise_if_issues(issues)
    return PreparedRegressionDataset(
        dataset=dataset,
        predictor_columns=predictor_columns,
        target_column=target_column,
        id_column=id_column,
        training_frame=training_frame,
        prediction_frame=prediction_frame,
        training_row_indices=tuple(training_row_indices),
        prediction_row_indices=tuple(prediction_row_indices),
        warnings=tuple(warnings),
    )



def prepare_matching_dataset(
    dataset: RegressionDataset,
    *,
    field: str = "dataset",
    min_rows: int = 4,
) -> PreparedMatchingDataset:
    issues = validate_role_counts(
        dataset,
        field=field,
        required_roles={
            DatasetColumnRole.TREATMENT: 1,
            DatasetColumnRole.OUTCOME: 1,
            DatasetColumnRole.PREDICTOR: 1,
        },
        maximum_roles={
            DatasetColumnRole.TREATMENT: 1,
            DatasetColumnRole.OUTCOME: 1,
        },
    )
    if issues:
        raise_if_issues(issues)

    predictor_columns = tuple(column for column in dataset.columns if column.role is DatasetColumnRole.PREDICTOR)
    treatment_column = next(column for column in dataset.columns if column.role is DatasetColumnRole.TREATMENT)
    outcome_column = next(column for column in dataset.columns if column.role is DatasetColumnRole.OUTCOME)
    id_column = next((column for column in dataset.columns if column.role is DatasetColumnRole.ID), None)

    treatment_index = dataset.columns.index(treatment_column)
    outcome_index = dataset.columns.index(outcome_column)
    predictor_indices = [dataset.columns.index(column) for column in predictor_columns]
    id_index = dataset.columns.index(id_column) if id_column is not None else None

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    treatment_labels: set[str] = set()

    for row_index, row in enumerate(dataset.rows):
        if _is_empty_row(row):
            warnings.append(f"Ignored empty row {row_index + 1}.")
            continue

        treatment_value = row.cells[treatment_index].strip()
        outcome_value = row.cells[outcome_index].strip()

        if not treatment_value:
            issues.append(build_error(f"{field}.rows.{row_index}.cells.{treatment_index}", f"{treatment_column.label} is required on row {row_index + 1}."))
            continue
        if not outcome_value:
            issues.append(build_error(f"{field}.rows.{row_index}.cells.{outcome_index}", f"{outcome_column.label} is required on row {row_index + 1}."))
            continue

        try:
            parsed_outcome = float(outcome_value)
        except ValueError:
            issues.append(build_error(f"{field}.rows.{row_index}.cells.{outcome_index}", f"{outcome_column.label} must be numeric on row {row_index + 1}."))
            continue

        predictor_values = _parse_numeric_predictors(
            row=row,
            row_index=row_index,
            field=field,
            predictor_columns=predictor_columns,
            predictor_indices=predictor_indices,
            issues=issues,
        )
        if predictor_values is None:
            continue

        treatment_labels.add(treatment_value)
        rows.append(
            {
                **predictor_values,
                treatment_column.key: treatment_value,
                outcome_column.key: parsed_outcome,
                "__row_id__": _row_identifier(row, row_index, id_index),
                "__row_index__": row_index,
            }
        )

    if len(rows) < min_rows:
        issues.append(build_error(f"{field}.rows", f"Provide at least {min_rows} populated rows for propensity score matching."))
    if len(treatment_labels) != 2:
        issues.append(build_error(f"{field}.rows", f"{treatment_column.label} must contain exactly two treatment groups."))

    dataframe = pd.DataFrame(rows)
    if not dataframe.empty and len(treatment_labels) == 2:
        counts = dataframe[treatment_column.key].value_counts()
        if (counts < 2).any():
            issues.append(build_error(f"{field}.rows", f"Each treatment group in {treatment_column.label} must contain at least two rows."))

    raise_if_issues(issues)
    return PreparedMatchingDataset(
        dataset=dataset,
        predictor_columns=predictor_columns,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        id_column=id_column,
        dataframe=dataframe,
        warnings=tuple(warnings),
    )



def require_binary_target(prepared: PreparedRegressionDataset, *, field: str = "dataset.rows") -> None:
    values = prepared.training_frame[prepared.target_column.key]
    class_count = int(values.nunique())
    if class_count != 2:
        raise_if_issues([build_error(field, f"{prepared.target_column.label} must contain exactly two classes in the training rows.")])



def require_multiclass_target(prepared: PreparedRegressionDataset, *, field: str = "dataset.rows") -> None:
    values = prepared.training_frame[prepared.target_column.key]
    class_count = int(values.nunique())
    if class_count < 3:
        raise_if_issues([build_error(field, f"{prepared.target_column.label} must contain at least three classes in the training rows.")])
