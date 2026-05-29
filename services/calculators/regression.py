from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import warnings as python_warnings
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score

from domain.enums import DatasetColumnRole, TestFamily, WorkflowKind
from domain.metadata import CalculatorMetadata
from domain.regression_inputs import MatchingPairResult, PredictionRowResult, PreparedMatchingDataset, PreparedRegressionDataset, RegressionColumn, RegressionDataset, RegressionRow
from domain.results import CalculationResult, DisplayValue, ResultMetric, ResultSection, ResultTable, display_number, format_number
from services.calculators.base import RegressionCalculator
from services.regression_validators import TARGET_KIND_CATEGORICAL, TARGET_KIND_NUMERIC, prepare_matching_dataset, prepare_supervised_dataset


ROLE_LABELS = {
    DatasetColumnRole.PREDICTOR: "Predictor",
    DatasetColumnRole.TARGET: "Target",
    DatasetColumnRole.TREATMENT: "Treatment",
    DatasetColumnRole.OUTCOME: "Outcome",
    DatasetColumnRole.ID: "ID",
    DatasetColumnRole.UNUSED: "Unused",
}



def _dataset(columns: list[tuple[str, DatasetColumnRole]], rows: list[list[str]]) -> dict[str, object]:
    return RegressionDataset(
        columns=tuple(
            RegressionColumn(key=f"column_{index + 1}", label=label, role=role)
            for index, (label, role) in enumerate(columns)
        ),
        rows=tuple(RegressionRow(cells=tuple(row)) for row in rows),
    ).to_dict()



def _blank_like(dataset: dict[str, object], *, rows: int) -> dict[str, object]:
    columns = dataset["columns"]
    return {
        "columns": columns,
        "rows": [{"cells": ["" for _ in columns]} for _ in range(rows)],
        "sourceMode": "grid",
        "filename": "",
    }



def _role_options(*roles: DatasetColumnRole) -> list[dict[str, str]]:
    return [{"value": role.value, "label": ROLE_LABELS[role]} for role in roles]



def _schema(
    *,
    default_dataset: dict[str, object],
    example_dataset: dict[str, object],
    blank_dataset: dict[str, object],
    role_options: list[dict[str, str]],
    allow_add_columns: bool,
    import_hint: str,
) -> dict[str, object]:
    return {
        "defaultDataset": default_dataset,
        "exampleDataset": example_dataset,
        "blankDataset": blank_dataset,
        "roleOptions": role_options,
        "allowAddColumns": allow_add_columns,
        "importHint": import_hint,
    }


SIMPLE_LINEAR_EXAMPLE = _dataset(
    [("Hours studied", DatasetColumnRole.PREDICTOR), ("Exam score", DatasetColumnRole.TARGET)],
    [
        ["1", "52"],
        ["2", "56"],
        ["3", "61"],
        ["4", "67"],
        ["5", "72"],
        ["6", "78"],
        ["7", ""],
        ["8", ""],
    ],
)
MULTIPLE_LINEAR_EXAMPLE = _dataset(
    [
        ("TV spend", DatasetColumnRole.PREDICTOR),
        ("Search spend", DatasetColumnRole.PREDICTOR),
        ("Social spend", DatasetColumnRole.PREDICTOR),
        ("Revenue", DatasetColumnRole.TARGET),
    ],
    [
        ["10", "4", "2", "110"],
        ["12", "5", "3", "121"],
        ["15", "6", "4", "138"],
        ["18", "7", "5", "151"],
        ["20", "8", "6", "165"],
        ["22", "9", "7", ""],
        ["24", "10", "8", ""],
    ],
)
BULK_LINEAR_EXAMPLE = _dataset(
    [
        ("Square footage", DatasetColumnRole.PREDICTOR),
        ("Bedrooms", DatasetColumnRole.PREDICTOR),
        ("Bathrooms", DatasetColumnRole.PREDICTOR),
        ("Sale price", DatasetColumnRole.TARGET),
    ],
    [
        ["900", "2", "1", "180"],
        ["1100", "2", "2", "215"],
        ["1400", "3", "2", "260"],
        ["1600", "3", "2", "295"],
        ["1850", "4", "3", "340"],
        ["2000", "4", "3", ""],
        ["2200", "4", "3", ""],
        ["2500", "5", "4", ""],
    ],
)
BINARY_LOGISTIC_EXAMPLE = _dataset(
    [
        ("Age", DatasetColumnRole.PREDICTOR),
        ("Income", DatasetColumnRole.PREDICTOR),
        ("Subscribed", DatasetColumnRole.TARGET),
    ],
    [
        ["24", "35", "No"],
        ["29", "40", "No"],
        ["35", "50", "Yes"],
        ["42", "62", "Yes"],
        ["48", "70", "Yes"],
        ["31", "44", "No"],
        ["38", "58", "Yes"],
        ["33", "47", ""],
        ["45", "68", ""],
    ],
)
MULTINOMIAL_LOGISTIC_EXAMPLE = _dataset(
    [
        ("Tenure", DatasetColumnRole.PREDICTOR),
        ("Usage", DatasetColumnRole.PREDICTOR),
        ("Plan", DatasetColumnRole.TARGET),
    ],
    [
        ["2", "10", "Basic"],
        ["4", "18", "Basic"],
        ["8", "30", "Standard"],
        ["10", "36", "Standard"],
        ["15", "55", "Premium"],
        ["18", "63", "Premium"],
        ["12", "41", "Standard"],
        ["6", "22", ""],
        ["16", "58", ""],
    ],
)
PSM_EXAMPLE = _dataset(
    [
        ("Record ID", DatasetColumnRole.ID),
        ("Age", DatasetColumnRole.PREDICTOR),
        ("Risk score", DatasetColumnRole.PREDICTOR),
        ("Treatment", DatasetColumnRole.TREATMENT),
        ("Outcome", DatasetColumnRole.OUTCOME),
    ],
    [
        ["T1", "44", "0.82", "Treated", "78"],
        ["T2", "39", "0.74", "Treated", "73"],
        ["T3", "47", "0.91", "Treated", "81"],
        ["T4", "41", "0.69", "Treated", "76"],
        ["C1", "43", "0.80", "Control", "69"],
        ["C2", "37", "0.71", "Control", "68"],
        ["C3", "49", "0.89", "Control", "72"],
        ["C4", "40", "0.67", "Control", "70"],
    ],
)

SIMPLE_LINEAR_SCHEMA = _schema(
    default_dataset=SIMPLE_LINEAR_EXAMPLE,
    example_dataset=SIMPLE_LINEAR_EXAMPLE,
    blank_dataset=_blank_like(SIMPLE_LINEAR_EXAMPLE, rows=6),
    role_options=_role_options(DatasetColumnRole.PREDICTOR, DatasetColumnRole.TARGET, DatasetColumnRole.ID, DatasetColumnRole.UNUSED),
    allow_add_columns=True,
    import_hint="Paste or import two columns: one predictor and one target. Leave the target blank on rows you want predicted.",
)
MULTIPLE_LINEAR_SCHEMA = _schema(
    default_dataset=MULTIPLE_LINEAR_EXAMPLE,
    example_dataset=MULTIPLE_LINEAR_EXAMPLE,
    blank_dataset=_blank_like(MULTIPLE_LINEAR_EXAMPLE, rows=6),
    role_options=_role_options(DatasetColumnRole.PREDICTOR, DatasetColumnRole.TARGET, DatasetColumnRole.ID, DatasetColumnRole.UNUSED),
    allow_add_columns=True,
    import_hint="Paste or import one target column plus two or more numeric predictor columns. Leave the target blank on rows to predict.",
)
BULK_LINEAR_SCHEMA = _schema(
    default_dataset=BULK_LINEAR_EXAMPLE,
    example_dataset=BULK_LINEAR_EXAMPLE,
    blank_dataset=_blank_like(BULK_LINEAR_EXAMPLE, rows=8),
    role_options=_role_options(DatasetColumnRole.PREDICTOR, DatasetColumnRole.TARGET, DatasetColumnRole.ID, DatasetColumnRole.UNUSED),
    allow_add_columns=True,
    import_hint="Train one linear model on completed rows, then predict many blank target rows in the same sheet.",
)
BINARY_LOGISTIC_SCHEMA = _schema(
    default_dataset=BINARY_LOGISTIC_EXAMPLE,
    example_dataset=BINARY_LOGISTIC_EXAMPLE,
    blank_dataset=_blank_like(BINARY_LOGISTIC_EXAMPLE, rows=7),
    role_options=_role_options(DatasetColumnRole.PREDICTOR, DatasetColumnRole.TARGET, DatasetColumnRole.ID, DatasetColumnRole.UNUSED),
    allow_add_columns=True,
    import_hint="Use numeric predictors and a two-class target. Leave the target blank on rows that need predicted classes and probabilities.",
)
MULTINOMIAL_LOGISTIC_SCHEMA = _schema(
    default_dataset=MULTINOMIAL_LOGISTIC_EXAMPLE,
    example_dataset=MULTINOMIAL_LOGISTIC_EXAMPLE,
    blank_dataset=_blank_like(MULTINOMIAL_LOGISTIC_EXAMPLE, rows=7),
    role_options=_role_options(DatasetColumnRole.PREDICTOR, DatasetColumnRole.TARGET, DatasetColumnRole.ID, DatasetColumnRole.UNUSED),
    allow_add_columns=True,
    import_hint="Use numeric predictors and a target with three or more classes. Leave target cells blank on rows to classify.",
)
PSM_SCHEMA = _schema(
    default_dataset=PSM_EXAMPLE,
    example_dataset=PSM_EXAMPLE,
    blank_dataset=_blank_like(PSM_EXAMPLE, rows=8),
    role_options=_role_options(DatasetColumnRole.ID, DatasetColumnRole.PREDICTOR, DatasetColumnRole.TREATMENT, DatasetColumnRole.OUTCOME, DatasetColumnRole.UNUSED),
    allow_add_columns=True,
    import_hint="Include predictors, exactly one treatment column, exactly one outcome column, and an optional ID column for matched-pair output.",
)



def _capture_fit_warnings(fit_callable) -> tuple[Any, tuple[str, ...]]:
    with python_warnings.catch_warnings(record=True) as records:
        python_warnings.simplefilter("always", ConvergenceWarning)
        fitted = fit_callable()
    warnings = tuple(str(record.message) for record in records if issubclass(record.category, ConvergenceWarning))
    return fitted, warnings



def _predictor_keys(prepared: PreparedRegressionDataset | PreparedMatchingDataset) -> list[str]:
    return [column.key for column in prepared.predictor_columns]



def _fit_linear_model(prepared: PreparedRegressionDataset) -> tuple[LinearRegression, float, float]:
    predictor_keys = _predictor_keys(prepared)
    target_key = prepared.target_column.key
    X_train = prepared.training_frame[predictor_keys].to_numpy(dtype=float)
    y_train = prepared.training_frame[target_key].to_numpy(dtype=float)
    model = LinearRegression().fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    return model, float(r2_score(y_train, train_predictions)), float(np.sqrt(mean_squared_error(y_train, train_predictions)))



def _build_linear_result(
    prepared: PreparedRegressionDataset,
    *,
    metadata: CalculatorMetadata,
    model: LinearRegression,
    r_squared: float,
    rmse: float,
    interpretation: str,
    notes: tuple[str, ...],
    metrics: tuple[ResultMetric, ...],
    require_prediction_rows: bool = False,
) -> CalculationResult:
    predictor_keys = _predictor_keys(prepared)
    predicted_values: np.ndarray | None = None
    if not prepared.prediction_frame.empty:
        X_predict = prepared.prediction_frame[predictor_keys].to_numpy(dtype=float)
        predicted_values = model.predict(X_predict)

    prediction_table = _prediction_table(prepared, predicted_values) if predicted_values is not None else None
    if require_prediction_rows and prediction_table is None:
        raise ValueError("Prediction rows are required for this calculator.")

    tables = (_linear_coefficients_table(prepared, model),)
    if prediction_table is not None:
        tables += (prediction_table,)

    return CalculationResult(
        slug=metadata.slug,
        test_name=metadata.name,
        statistic_name="R²",
        statistic=display_number(r_squared),
        metrics=metrics + (ResultMetric("RMSE", format_number(rmse), emphasis=True),),
        tables=tables,
        interpretation=interpretation,
        warnings=prepared.warnings,
        notes=notes,
        dataset=_filled_dataset(prepared, predicted_values) if predicted_values is not None else None,
    )



def _fit_logistic_model(
    prepared: PreparedRegressionDataset,
    *,
    max_iter: int,
    multi_class: str | None = None,
) -> tuple[LogisticRegression, float, float, tuple[str, ...]]:
    predictor_keys = _predictor_keys(prepared)
    target_key = prepared.target_column.key
    X_train = prepared.training_frame[predictor_keys].to_numpy(dtype=float)
    y_train = prepared.training_frame[target_key].astype(str).to_numpy()
    model_factory = lambda: LogisticRegression(max_iter=max_iter, **({"multi_class": multi_class} if multi_class else {})).fit(X_train, y_train)
    model, fit_warnings = _capture_fit_warnings(model_factory)
    train_probabilities = model.predict_proba(X_train)
    train_predictions = model.predict(X_train)
    accuracy = float(accuracy_score(y_train, train_predictions))
    loss = float(log_loss(y_train, train_probabilities, labels=model.classes_))
    return model, accuracy, loss, fit_warnings



def _build_logistic_result(
    prepared: PreparedRegressionDataset,
    *,
    metadata: CalculatorMetadata,
    model: LogisticRegression,
    accuracy: float,
    loss: float,
    fit_warnings: tuple[str, ...],
    interpretation: str,
    notes: tuple[str, ...],
    metrics: tuple[ResultMetric, ...],
) -> CalculationResult:
    predictor_keys = _predictor_keys(prepared)
    predicted_labels: np.ndarray | None = None
    predicted_probabilities: np.ndarray | None = None
    if not prepared.prediction_frame.empty:
        X_predict = prepared.prediction_frame[predictor_keys].to_numpy(dtype=float)
        predicted_labels = model.predict(X_predict)
        predicted_probabilities = model.predict_proba(X_predict)

    prediction_table = (
        _classification_prediction_table(prepared, predicted_labels, predicted_probabilities, model.classes_)
        if predicted_labels is not None and predicted_probabilities is not None
        else None
    )
    tables = (_logistic_coefficients_table(prepared, model),)
    if prediction_table is not None:
        tables += (prediction_table,)

    return CalculationResult(
        slug=metadata.slug,
        test_name=metadata.name,
        statistic_name="Accuracy",
        statistic=display_number(accuracy),
        metrics=metrics + (ResultMetric("Log loss", format_number(loss), emphasis=True),),
        tables=tables,
        interpretation=interpretation,
        warnings=tuple((*prepared.warnings, *_class_balance_warning(prepared), *fit_warnings)),
        notes=notes,
        dataset=_filled_dataset_with_labels(prepared, predicted_labels) if predicted_labels is not None else None,
    )



def _nearest_available_control_index(control_scores: np.ndarray, available_indices: list[int], treated_score: float) -> tuple[int, float]:
    available_scores = control_scores[available_indices]
    nearest_position = int(np.argmin(np.abs(available_scores - treated_score)))
    matched_control_index = available_indices.pop(nearest_position)
    return matched_control_index, float(abs(control_scores[matched_control_index] - treated_score))



def _filled_dataset(prepared: PreparedRegressionDataset, predicted_values: np.ndarray) -> dict[str, object]:
    target_index = prepared.dataset.columns.index(prepared.target_column)
    replacements = {
        (row_index, target_index): format_number(float(predicted_value))
        for row_index, predicted_value in zip(prepared.prediction_row_indices, predicted_values, strict=True)
    }
    return prepared.dataset.with_filled_cells(replacements).to_dict()



def _filled_dataset_with_labels(prepared: PreparedRegressionDataset, predicted_labels: np.ndarray) -> dict[str, object]:
    target_index = prepared.dataset.columns.index(prepared.target_column)
    replacements = {
        (row_index, target_index): str(predicted_label)
        for row_index, predicted_label in zip(prepared.prediction_row_indices, predicted_labels, strict=True)
    }
    return prepared.dataset.with_filled_cells(replacements).to_dict()



def _prediction_table(prepared: PreparedRegressionDataset, predicted_values: np.ndarray) -> ResultTable | None:
    if not prepared.prediction_row_indices:
        return None
    rows = []
    target_index = prepared.dataset.columns.index(prepared.target_column)
    for row_index, predicted_value in zip(prepared.prediction_row_indices, predicted_values, strict=True):
        original_cells = list(prepared.dataset.rows[row_index].cells)
        original_cells[target_index] = format_number(float(predicted_value))
        rows.append(PredictionRowResult(row_number=row_index + 1, values=tuple(original_cells)).to_table_row())
    return ResultTable(
        title="Predicted rows",
        columns=("Row", *(column.label for column in prepared.dataset.columns)),
        rows=tuple(rows),
        caption="Rows with blank target cells are filled after the model is fitted on the completed training rows.",
    )



def _classification_prediction_table(
    prepared: PreparedRegressionDataset,
    predicted_labels: np.ndarray,
    predicted_probabilities: np.ndarray,
    classes: np.ndarray,
) -> ResultTable | None:
    if not prepared.prediction_row_indices:
        return None

    rows: list[tuple[str, ...]] = []
    target_index = prepared.dataset.columns.index(prepared.target_column)
    for row_index, predicted_label, probability_row in zip(
        prepared.prediction_row_indices,
        predicted_labels,
        predicted_probabilities,
        strict=True,
    ):
        original_cells = list(prepared.dataset.rows[row_index].cells)
        original_cells[target_index] = str(predicted_label)
        probability_cells = tuple(format_number(float(probability)) for probability in probability_row)
        rows.append(PredictionRowResult(row_number=row_index + 1, values=tuple((*original_cells, *probability_cells))).to_table_row())

    return ResultTable(
        title="Predicted rows",
        columns=(
            "Row",
            *(column.label for column in prepared.dataset.columns),
            *(f"P({class_label})" for class_label in classes),
        ),
        rows=tuple(rows),
        caption="Blank target rows are classified after fitting the model on the labeled training rows.",
    )



def _linear_coefficients_table(prepared: PreparedRegressionDataset, model: LinearRegression) -> ResultTable:
    rows = [("Intercept", format_number(float(model.intercept_)))]
    rows.extend(
        (column.label, format_number(float(coefficient)))
        for column, coefficient in zip(prepared.predictor_columns, model.coef_, strict=True)
    )
    return ResultTable(
        title="Coefficients",
        columns=("Term", "Coefficient"),
        rows=tuple(rows),
    )



def _logistic_coefficients_table(prepared: PreparedRegressionDataset, model: LogisticRegression) -> ResultTable:
    if len(model.classes_) == 2:
        positive_class = str(model.classes_[1])
        rows = [(positive_class, "Intercept", format_number(float(model.intercept_[0])), format_number(float(np.exp(model.intercept_[0]))))]
        rows.extend(
            (
                positive_class,
                column.label,
                format_number(float(coefficient)),
                format_number(float(np.exp(coefficient))),
            )
            for column, coefficient in zip(prepared.predictor_columns, model.coef_[0], strict=True)
        )
    else:
        rows = []
        for class_label, intercept, coefficients in zip(model.classes_, model.intercept_, model.coef_, strict=True):
            rows.append((str(class_label), "Intercept", format_number(float(intercept)), format_number(float(np.exp(intercept)))))
            rows.extend(
                (
                    str(class_label),
                    column.label,
                    format_number(float(coefficient)),
                    format_number(float(np.exp(coefficient))),
                )
                for column, coefficient in zip(prepared.predictor_columns, coefficients, strict=True)
            )
    return ResultTable(
        title="Coefficients",
        columns=("Class", "Term", "Log-odds coefficient", "Odds ratio"),
        rows=tuple(rows),
    )



def _class_balance_warning(prepared: PreparedRegressionDataset) -> tuple[str, ...]:
    counts = Counter(str(value) for value in prepared.training_frame[prepared.target_column.key])
    total = sum(counts.values())
    if not counts or total == 0:
        return ()
    minority_share = min(counts.values()) / total
    if minority_share < 0.2:
        return ("Training classes are imbalanced, so probability estimates may be unstable.",)
    return ()



def _select_treated_and_control_labels(values: np.ndarray) -> tuple[str, str]:
    labels = [str(value) for value in values]
    lower_map = {label.casefold(): label for label in labels}
    for preferred in ("treated", "treatment", "yes", "1"):
        if preferred in lower_map:
            treated = lower_map[preferred]
            control = next(label for label in labels if label != treated)
            return treated, control
    ordered = sorted(labels)
    return ordered[-1], ordered[0]



def _standardized_mean_difference(treated: np.ndarray, control: np.ndarray) -> float:
    treated_mean = float(np.mean(treated))
    control_mean = float(np.mean(control))
    treated_var = float(np.var(treated, ddof=1)) if len(treated) > 1 else 0.0
    control_var = float(np.var(control, ddof=1)) if len(control) > 1 else 0.0
    pooled = np.sqrt((treated_var + control_var) / 2)
    if pooled == 0:
        return 0.0
    return (treated_mean - control_mean) / pooled



def _balance_table(prepared: PreparedMatchingDataset, treated_frame, control_frame, matched_pairs: tuple[MatchingPairResult, ...]) -> ResultTable:
    rows = []
    if matched_pairs:
        matched_treated_ids = [pair.treated_id for pair in matched_pairs]
        matched_control_ids = [pair.control_id for pair in matched_pairs]
        matched_treated = treated_frame[treated_frame["__row_id__"].isin(matched_treated_ids)]
        matched_control = control_frame[control_frame["__row_id__"].isin(matched_control_ids)]
    else:
        matched_treated = treated_frame.iloc[0:0]
        matched_control = control_frame.iloc[0:0]

    for predictor_column in prepared.predictor_columns:
        before = _standardized_mean_difference(
            treated_frame[predictor_column.key].to_numpy(dtype=float),
            control_frame[predictor_column.key].to_numpy(dtype=float),
        )
        after = _standardized_mean_difference(
            matched_treated[predictor_column.key].to_numpy(dtype=float) if not matched_treated.empty else np.array([0.0]),
            matched_control[predictor_column.key].to_numpy(dtype=float) if not matched_control.empty else np.array([0.0]),
        )
        rows.append((predictor_column.label, format_number(before), format_number(after)))

    return ResultTable(
        title="Balance summary",
        columns=("Predictor", "SMD before matching", "SMD after matching"),
        rows=tuple(rows),
        caption="Standardized mean differences closer to 0 indicate better balance between treatment groups.",
    )


class SimpleLinearRegressionCalculator(RegressionCalculator):
    metadata = CalculatorMetadata(
        catalog_position=1,
        slug="simple-linear-regression",
        name="Simple Linear Regression",
        family=TestFamily.REGRESSION,
        description="Fit one linear predictor to a numeric outcome and fill blank target rows with predicted values.",
        check="How strongly one numeric predictor explains a continuous target and what values the fitted line predicts for blank target rows.",
        statistic_formula="ŷ = β₀ + β₁x",
        assumptions=(
            "The predictor and target have an approximately linear relationship.",
            "Training rows are independent of each other.",
            "The target is continuous and measured on a comparable scale.",
        ),
        required_sample_data=(
            "Exactly one predictor column and one target column.",
            "At least three completed training rows.",
            "Optional blank target rows for prediction.",
        ),
        section_slug="regression",
        workflow_kind=WorkflowKind.DATASET,
        dataset_schema=SIMPLE_LINEAR_SCHEMA,
        extra_default_values={"dataset": SIMPLE_LINEAR_SCHEMA["defaultDataset"]},
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> PreparedRegressionDataset:
        dataset = self.parse_dataset(raw_data)
        return prepare_supervised_dataset(
            dataset,
            min_training_rows=3,
            min_predictor_columns=1,
            max_predictor_columns=1,
            target_kind=TARGET_KIND_NUMERIC,
        )

    def calculate_result(self, normalized_input: PreparedRegressionDataset) -> CalculationResult:
        model, r_squared, rmse = _fit_linear_model(normalized_input)
        return _build_linear_result(
            normalized_input,
            metadata=self.metadata,
            model=model,
            r_squared=r_squared,
            rmse=rmse,
            interpretation="The fitted line summarizes the linear relationship between the predictor and the continuous target, then fills any blank target rows with predicted values.",
            notes=("Rows with blank target cells are excluded from model fitting and used only for prediction.",),
            metrics=(
                ResultMetric("Training rows", str(len(normalized_input.training_row_indices))),
                ResultMetric("Prediction rows", str(len(normalized_input.prediction_row_indices))),
                ResultMetric("Intercept", format_number(float(model.intercept_))),
            ),
        )


class MultipleLinearRegressionCalculator(RegressionCalculator):
    metadata = CalculatorMetadata(
        catalog_position=2,
        slug="multiple-linear-regression",
        name="Multiple Linear Regression",
        family=TestFamily.REGRESSION,
        description="Fit several numeric predictors to one continuous target and fill blank target rows after training.",
        check="How multiple numeric predictors jointly explain a continuous target and what they predict for incomplete rows.",
        statistic_formula="ŷ = β₀ + β₁x₁ + β₂x₂ + … + βₖxₖ",
        assumptions=(
            "Predictors relate approximately linearly to the target.",
            "Training rows are independent.",
            "Predictors are numeric and measured consistently across rows.",
        ),
        required_sample_data=(
            "Two or more predictor columns and one target column.",
            "At least four completed training rows.",
            "Optional blank target rows for prediction.",
        ),
        section_slug="regression",
        workflow_kind=WorkflowKind.DATASET,
        dataset_schema=MULTIPLE_LINEAR_SCHEMA,
        extra_default_values={"dataset": MULTIPLE_LINEAR_SCHEMA["defaultDataset"]},
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> PreparedRegressionDataset:
        dataset = self.parse_dataset(raw_data)
        return prepare_supervised_dataset(
            dataset,
            min_training_rows=4,
            min_predictor_columns=2,
            target_kind=TARGET_KIND_NUMERIC,
        )

    def calculate_result(self, normalized_input: PreparedRegressionDataset) -> CalculationResult:
        model, r_squared, rmse = _fit_linear_model(normalized_input)
        return _build_linear_result(
            normalized_input,
            metadata=self.metadata,
            model=model,
            r_squared=r_squared,
            rmse=rmse,
            interpretation="The fitted model combines all predictor columns to estimate the continuous target and populate blank target cells in prediction rows.",
            notes=("All predictor columns are treated as numeric inputs in the design matrix.",),
            metrics=(
                ResultMetric("Predictors", str(len(normalized_input.predictor_columns))),
                ResultMetric("Training rows", str(len(normalized_input.training_row_indices))),
                ResultMetric("Prediction rows", str(len(normalized_input.prediction_row_indices))),
            ),
        )


class BulkLinearRegressionCalculator(RegressionCalculator):
    metadata = CalculatorMetadata(
        catalog_position=3,
        slug="bulk-linear-regression",
        name="Bulk Linear Regression",
        family=TestFamily.REGRESSION,
        description="Train one linear regression model once and score many blank target rows in a single spreadsheet-style run.",
        check="What one fitted linear model predicts for a batch of rows where the target is intentionally left blank.",
        statistic_formula="ŷ = β₀ + Σβᵢxᵢ for many prediction rows",
        assumptions=(
            "Training rows reflect a stable linear relationship.",
            "Predictor columns are numeric.",
            "Blank target rows are used only for scoring, not training.",
        ),
        required_sample_data=(
            "One target column with completed training rows.",
            "One or more numeric predictor columns.",
            "At least two blank target rows to score in bulk.",
        ),
        section_slug="regression",
        workflow_kind=WorkflowKind.DATASET,
        dataset_schema=BULK_LINEAR_SCHEMA,
        extra_default_values={"dataset": BULK_LINEAR_SCHEMA["defaultDataset"]},
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> PreparedRegressionDataset:
        dataset = self.parse_dataset(raw_data)
        return prepare_supervised_dataset(
            dataset,
            min_training_rows=4,
            min_prediction_rows=2,
            min_predictor_columns=1,
            target_kind=TARGET_KIND_NUMERIC,
        )

    def calculate_result(self, normalized_input: PreparedRegressionDataset) -> CalculationResult:
        model, r_squared, rmse = _fit_linear_model(normalized_input)
        return _build_linear_result(
            normalized_input,
            metadata=self.metadata,
            model=model,
            r_squared=r_squared,
            rmse=rmse,
            interpretation="One linear model was fitted on the completed rows and then applied across the batch of blank target rows.",
            notes=("Bulk linear regression uses the same fitted model for every blank target row in the sheet.",),
            metrics=(
                ResultMetric("Predictors", str(len(normalized_input.predictor_columns))),
                ResultMetric("Training rows", str(len(normalized_input.training_row_indices))),
                ResultMetric("Predicted rows", str(len(normalized_input.prediction_row_indices))),
            ),
            require_prediction_rows=True,
        )


class BinaryLogisticRegressionCalculator(RegressionCalculator):
    metadata = CalculatorMetadata(
        catalog_position=4,
        slug="binary-logistic-regression",
        name="Binary Logistic Regression",
        family=TestFamily.CLASSIFICATION,
        description="Fit a two-class logistic model, then fill blank target rows with predicted labels and class probabilities.",
        check="How numeric predictors separate two target classes and what class probabilities they imply for unlabeled rows.",
        statistic_formula="logit(P(Y=1)) = β₀ + Σβᵢxᵢ",
        assumptions=(
            "Predictors are numeric and measured consistently.",
            "Training rows represent exactly two classes.",
            "Blank target rows are reserved for prediction only.",
        ),
        required_sample_data=(
            "One target column with exactly two classes in the training rows.",
            "One or more numeric predictor columns.",
            "Optional blank target rows for prediction.",
        ),
        section_slug="regression",
        workflow_kind=WorkflowKind.DATASET,
        dataset_schema=BINARY_LOGISTIC_SCHEMA,
        extra_default_values={"dataset": BINARY_LOGISTIC_SCHEMA["defaultDataset"]},
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> PreparedRegressionDataset:
        dataset = self.parse_dataset(raw_data)
        return prepare_supervised_dataset(
            dataset,
            min_training_rows=4,
            min_predictor_columns=1,
            target_kind=TARGET_KIND_CATEGORICAL,
            minimum_classes=2,
            maximum_classes=2,
        )

    def calculate_result(self, normalized_input: PreparedRegressionDataset) -> CalculationResult:
        model, accuracy, loss, fit_warnings = _fit_logistic_model(normalized_input, max_iter=2000)
        return _build_logistic_result(
            normalized_input,
            metadata=self.metadata,
            model=model,
            accuracy=accuracy,
            loss=loss,
            fit_warnings=fit_warnings,
            interpretation="The logistic model estimates class probabilities for each row and fills blank targets with the most likely class.",
            notes=("Probabilities are reported for every class so you can inspect prediction confidence.",),
            metrics=(
                ResultMetric("Training rows", str(len(normalized_input.training_row_indices))),
                ResultMetric("Prediction rows", str(len(normalized_input.prediction_row_indices))),
                ResultMetric("Classes", ", ".join(str(class_label) for class_label in model.classes_)),
            ),
        )


class MultinomialLogisticRegressionCalculator(RegressionCalculator):
    metadata = CalculatorMetadata(
        catalog_position=5,
        slug="multinomial-logistic-regression",
        name="Multinomial Logistic Regression",
        family=TestFamily.CLASSIFICATION,
        description="Fit a multi-class logistic model and fill blank target rows with predicted labels and per-class probabilities.",
        check="How numeric predictors explain a target with three or more categories and which class is most probable for unlabeled rows.",
        statistic_formula="P(Y=c) = softmax(β₀,c + Σβᵢ,c xᵢ)",
        assumptions=(
            "Predictors are numeric.",
            "Training rows contain at least three target classes.",
            "Blank target rows are held back for prediction.",
        ),
        required_sample_data=(
            "One target column with at least three classes in the training rows.",
            "One or more numeric predictor columns.",
            "Optional blank target rows for prediction.",
        ),
        section_slug="regression",
        workflow_kind=WorkflowKind.DATASET,
        dataset_schema=MULTINOMIAL_LOGISTIC_SCHEMA,
        extra_default_values={"dataset": MULTINOMIAL_LOGISTIC_SCHEMA["defaultDataset"]},
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> PreparedRegressionDataset:
        dataset = self.parse_dataset(raw_data)
        return prepare_supervised_dataset(
            dataset,
            min_training_rows=6,
            min_predictor_columns=1,
            target_kind=TARGET_KIND_CATEGORICAL,
            minimum_classes=3,
        )

    def calculate_result(self, normalized_input: PreparedRegressionDataset) -> CalculationResult:
        model, accuracy, loss, fit_warnings = _fit_logistic_model(normalized_input, max_iter=3000, multi_class="multinomial")
        return _build_logistic_result(
            normalized_input,
            metadata=self.metadata,
            model=model,
            accuracy=accuracy,
            loss=loss,
            fit_warnings=fit_warnings,
            interpretation="The multinomial logistic model estimates probabilities for every class and assigns the highest-probability class to each blank target row.",
            notes=("Per-class coefficient rows show how each predictor changes the relative log-odds for each class.",),
            metrics=(
                ResultMetric("Training rows", str(len(normalized_input.training_row_indices))),
                ResultMetric("Prediction rows", str(len(normalized_input.prediction_row_indices))),
                ResultMetric("Classes", str(len(model.classes_))),
            ),
        )


class PropensityScoreMatchingCalculator(RegressionCalculator):
    metadata = CalculatorMetadata(
        catalog_position=6,
        slug="propensity-score-matching",
        name="Propensity Score Matching",
        family=TestFamily.CAUSAL_INFERENCE,
        description="Estimate propensity scores and perform 1:1 nearest-neighbor matching without replacement.",
        check="How well treatment and control rows can be matched on observed predictors before comparing their outcomes.",
        statistic_formula="ATT = mean(Yᵗʳᵉᵃᵗᵉᵈ - Yᶜᵒⁿᵗʳᵒˡ) after 1:1 nearest-neighbor matching",
        assumptions=(
            "Predictors capture the main observed confounders.",
            "Treatment assignment has exactly two groups.",
            "Outcome values are numeric and measured on a comparable scale.",
        ),
        required_sample_data=(
            "One treatment column with exactly two groups.",
            "One numeric outcome column.",
            "One or more numeric predictor columns and an optional ID column.",
        ),
        section_slug="regression",
        workflow_kind=WorkflowKind.DATASET,
        dataset_schema=PSM_SCHEMA,
        extra_default_values={"dataset": PSM_SCHEMA["defaultDataset"]},
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> PreparedMatchingDataset:
        dataset = self.parse_dataset(raw_data)
        return prepare_matching_dataset(dataset)

    def calculate_result(self, normalized_input: PreparedMatchingDataset) -> CalculationResult:
        predictor_keys = _predictor_keys(normalized_input)
        treatment_key = normalized_input.treatment_column.key
        outcome_key = normalized_input.outcome_column.key
        treatment_series = normalized_input.dataframe[treatment_key].astype(str)
        treatment_labels = np.array(sorted(treatment_series.unique()))
        treated_label, control_label = _select_treated_and_control_labels(treatment_labels)
        treated_mask = treatment_series == treated_label
        X = normalized_input.dataframe[predictor_keys].to_numpy(dtype=float)
        y = treated_mask.astype(int).to_numpy()
        model, fit_warnings = _capture_fit_warnings(lambda: LogisticRegression(max_iter=2000).fit(X, y))
        dataframe = normalized_input.dataframe.assign(propensity_score=model.predict_proba(X)[:, 1])

        treated_frame = dataframe.loc[treated_mask].sort_values("propensity_score").reset_index(drop=True)
        control_frame = dataframe.loc[~treated_mask].reset_index(drop=True)
        control_scores = control_frame["propensity_score"].to_numpy(dtype=float)
        available_control_positions = list(range(len(control_frame)))
        matched_pairs: list[MatchingPairResult] = []
        unmatched_treated = 0

        for _, treated_row in treated_frame.iterrows():
            if not available_control_positions:
                unmatched_treated += 1
                continue
            matched_control_position, distance = _nearest_available_control_index(
                control_scores,
                available_control_positions,
                float(treated_row["propensity_score"]),
            )
            control_row = control_frame.iloc[matched_control_position]
            matched_pairs.append(
                MatchingPairResult(
                    treated_id=str(treated_row["__row_id__"]),
                    control_id=str(control_row["__row_id__"]),
                    treated_score=float(treated_row["propensity_score"]),
                    control_score=float(control_row["propensity_score"]),
                    distance=distance,
                    treated_outcome=float(treated_row[outcome_key]),
                    control_outcome=float(control_row[outcome_key]),
                )
            )

        pair_count = len(matched_pairs)
        att = float(np.mean([pair.treated_outcome - pair.control_outcome for pair in matched_pairs])) if matched_pairs else 0.0
        average_distance = float(np.mean([pair.distance for pair in matched_pairs])) if matched_pairs else 0.0

        pair_table = ResultTable(
            title="Matched pairs",
            columns=(
                "Treated ID",
                "Control ID",
                "Treated score",
                "Control score",
                "Distance",
                "Treated outcome",
                "Control outcome",
                "Outcome difference",
            ),
            rows=tuple(pair.to_table_row() for pair in matched_pairs),
            caption="Pairs are built with 1:1 nearest-neighbor matching without replacement on the estimated propensity score.",
        )
        balance_table = _balance_table(normalized_input, treated_frame, control_frame, tuple(matched_pairs))
        coefficient_table = ResultTable(
            title="Propensity model coefficients",
            columns=("Term", "Coefficient"),
            rows=tuple(
                [("Intercept", format_number(float(model.intercept_[0])))]
                + [(column.label, format_number(float(coefficient))) for column, coefficient in zip(normalized_input.predictor_columns, model.coef_[0], strict=True)]
            ),
        )

        warnings = [*normalized_input.warnings, *fit_warnings]
        if unmatched_treated:
            warnings.append(f"{unmatched_treated} treated row(s) could not be matched because no control rows remained.")

        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="Matched pairs",
            statistic=DisplayValue(raw=pair_count, display=str(pair_count)),
            metrics=(
                ResultMetric("Treated label", treated_label),
                ResultMetric("Control label", control_label),
                ResultMetric("ATT", format_number(att), emphasis=True),
                ResultMetric("Mean propensity gap", format_number(average_distance)),
            ),
            sections=(
                ResultSection(
                    title="Matched sample",
                    metrics=(
                        ResultMetric("Treated rows", str(len(treated_frame))),
                        ResultMetric("Control rows", str(len(control_frame))),
                        ResultMetric("Matched treated rows", str(pair_count)),
                        ResultMetric("Unmatched treated rows", str(unmatched_treated)),
                    ),
                ),
            ),
            tables=(coefficient_table, pair_table, balance_table),
            interpretation="The propensity model estimates each row's treatment probability, then greedily matches each treated row to the nearest remaining control row without replacement.",
            warnings=tuple(warnings),
            notes=("ATT is reported as the mean treated-minus-control outcome difference across the matched pairs.",),
        )
