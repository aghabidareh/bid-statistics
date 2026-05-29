from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from math import isclose
from typing import Any

from domain.enums import AlternativeHypothesis
from domain.inputs import ManovaRow, PairedRocObservation, RepeatedMeasureRow, RocObservation, SurvivalObservation, TwoWayAnovaRow
from domain.results import ValidationIssue


class ValidationIssues(Exception):
    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__("Validation failed")


ALTERNATIVE_VALUES = {choice.value for choice in AlternativeHypothesis}
KS_DISTRIBUTIONS = {"norm", "expon", "uniform"}


def raise_if_issues(issues: list[ValidationIssue]) -> None:
    if issues:
        raise ValidationIssues(issues)



def build_error(field: str, message: str) -> ValidationIssue:
    return ValidationIssue(field=field, message=message)



def errors_by_field(issues: Iterable[ValidationIssue]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for issue in issues:
        grouped[issue.field].append(issue.message)
    return dict(grouped)



def parse_float(value: object, field: str, label: str, issues: list[ValidationIssue]) -> float | None:
    raw_value = "" if value is None else str(value).strip()
    if not raw_value:
        issues.append(build_error(field, f"{label} is required."))
        return None
    try:
        return float(raw_value)
    except ValueError:
        issues.append(build_error(field, f"{label} must be a number."))
        return None



def parse_int(value: object, field: str, label: str, issues: list[ValidationIssue]) -> int | None:
    raw_value = "" if value is None else str(value).strip()
    if not raw_value:
        issues.append(build_error(field, f"{label} is required."))
        return None
    try:
        parsed = int(raw_value)
    except ValueError:
        issues.append(build_error(field, f"{label} must be an integer."))
        return None
    return parsed



def parse_positive_float(value: object, field: str, label: str, issues: list[ValidationIssue]) -> float | None:
    parsed = parse_float(value, field, label, issues)
    if parsed is not None and parsed <= 0:
        issues.append(build_error(field, f"{label} must be greater than zero."))
        return None
    return parsed



def parse_probability(value: object, field: str, label: str, issues: list[ValidationIssue]) -> float | None:
    parsed = parse_float(value, field, label, issues)
    if parsed is not None and not 0 <= parsed <= 1:
        issues.append(build_error(field, f"{label} must be between 0 and 1."))
        return None
    return parsed



def parse_alpha(value: object, field: str = "alpha") -> tuple[float | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    alpha = parse_float(value, field, "Alpha", issues)
    if alpha is not None and not 0 < alpha < 1:
        issues.append(build_error(field, "Alpha must be between 0 and 1."))
    return alpha, issues



def parse_alternative(value: object, field: str = "alternative") -> tuple[AlternativeHypothesis | None, list[ValidationIssue]]:
    raw_value = "" if value is None else str(value).strip()
    if raw_value not in ALTERNATIVE_VALUES:
        return None, [build_error(field, "Choose a valid alternative hypothesis.")]
    return AlternativeHypothesis(raw_value), []



def _clean_numeric_tokens(raw_value: str) -> list[str]:
    tokens = [token.strip() for token in raw_value.replace("\n", ",").split(",")]
    return [token for token in tokens if token]



def parse_numeric_series(
    value: object,
    field: str,
    label: str,
    *,
    minimum_length: int = 2,
    positive_only: bool = False,
    nonnegative: bool = False,
) -> tuple[tuple[float, ...] | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    raw_value = "" if value is None else str(value).strip()
    if not raw_value:
        issues.append(build_error(field, f"{label} is required."))
        return None, issues

    cleaned_tokens = _clean_numeric_tokens(raw_value)
    if not cleaned_tokens:
        issues.append(build_error(field, f"{label} must contain at least {minimum_length} numeric values."))
        return None, issues

    values: list[float] = []
    for token in cleaned_tokens:
        try:
            parsed = float(token)
        except ValueError:
            issues.append(build_error(field, f"{label} contains a non-numeric value: {token}."))
            continue
        if positive_only and parsed <= 0:
            issues.append(build_error(field, f"{label} must contain only positive values."))
            continue
        if nonnegative and parsed < 0:
            issues.append(build_error(field, f"{label} cannot include negative values."))
            continue
        values.append(parsed)

    if len(values) < minimum_length:
        issues.append(build_error(field, f"{label} must contain at least {minimum_length} numeric values."))

    return tuple(values) if not issues else None, issues



def parse_paired_numeric_samples(
    value_a: object,
    value_b: object,
    *,
    field_a: str,
    field_b: str,
    label_a: str,
    label_b: str,
    minimum_length: int = 2,
) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    sample_a, sample_a_issues = parse_numeric_series(value_a, field_a, label_a, minimum_length=minimum_length)
    sample_b, sample_b_issues = parse_numeric_series(value_b, field_b, label_b, minimum_length=minimum_length)
    issues.extend(sample_a_issues)
    issues.extend(sample_b_issues)
    if sample_a is not None and sample_b is not None and len(sample_a) != len(sample_b):
        issues.append(build_error(field_b, f"{label_a} and {label_b} must have the same length."))
    return sample_a, sample_b, issues



def parse_named_groups(
    value: object,
    field: str,
    *,
    minimum_groups: int = 2,
    minimum_length: int = 2,
) -> tuple[tuple[tuple[str, tuple[float, ...]], ...] | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    raw_value = "" if value is None else str(value).strip()
    if not raw_value:
        issues.append(build_error(field, "Group samples are required."))
        return None, issues

    groups: list[tuple[str, tuple[float, ...]]] = []
    seen_names: set[str] = set()
    for index, line in enumerate([line.strip() for line in raw_value.splitlines() if line.strip()], start=1):
        separator = ":" if ":" in line else "=" if "=" in line else None
        if separator is None:
            issues.append(build_error(field, f"Line {index} must look like 'Group A: 1, 2, 3'."))
            continue
        group_name, series = [part.strip() for part in line.split(separator, 1)]
        if not group_name:
            issues.append(build_error(field, f"Line {index} must include a group name."))
            continue
        if group_name in seen_names:
            issues.append(build_error(field, f"Group name '{group_name}' is duplicated."))
            continue
        seen_names.add(group_name)

        values, value_issues = parse_numeric_series(series, field, group_name, minimum_length=minimum_length)
        issues.extend(value_issues)
        if values is not None:
            groups.append((group_name, values))

    if len(groups) < minimum_groups:
        issues.append(build_error(field, f"Provide at least {minimum_groups} groups."))

    return tuple(groups) if not issues else None, issues



def _parse_rows(raw_value: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in raw_value.splitlines():
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        rows.append([token.strip() for token in cleaned_line.split(",")])
    return rows



def parse_repeated_measures_rows(value: object, field: str) -> tuple[tuple[RepeatedMeasureRow, ...] | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    raw_value = "" if value is None else str(value).strip()
    if not raw_value:
        issues.append(build_error(field, "Repeated-measures rows are required."))
        return None, issues

    rows: list[RepeatedMeasureRow] = []
    subjects: set[str] = set()
    conditions: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for index, columns in enumerate(_parse_rows(raw_value), start=1):
        if len(columns) != 3:
            issues.append(build_error(field, f"Line {index} must contain subject, condition, and value."))
            continue
        subject, condition, value_token = columns
        if not subject or not condition:
            issues.append(build_error(field, f"Line {index} must include both a subject and a condition."))
            continue
        try:
            parsed_value = float(value_token)
        except ValueError:
            issues.append(build_error(field, f"Line {index} contains a non-numeric value: {value_token}."))
            continue

        key = (subject, condition)
        if key in seen_pairs:
            issues.append(build_error(field, f"The subject/condition pair '{subject}, {condition}' is duplicated."))
            continue
        seen_pairs.add(key)
        subjects.add(subject)
        conditions.add(condition)
        rows.append(RepeatedMeasureRow(subject=subject, condition=condition, value=parsed_value))

    if len(subjects) < 2:
        issues.append(build_error(field, "Provide at least two subjects."))
    if len(conditions) < 2:
        issues.append(build_error(field, "Provide at least two conditions."))

    if rows and subjects and conditions:
        expected_pairs = {(subject, condition) for subject in subjects for condition in conditions}
        missing_pairs = expected_pairs.difference(seen_pairs)
        if missing_pairs:
            issues.append(build_error(field, "Each subject must have exactly one observation for every condition."))

    return tuple(rows) if not issues else None, issues



def parse_two_way_rows(value: object, field: str) -> tuple[tuple[TwoWayAnovaRow, ...] | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    raw_value = "" if value is None else str(value).strip()
    if not raw_value:
        issues.append(build_error(field, "Two-way ANOVA rows are required."))
        return None, issues

    rows: list[TwoWayAnovaRow] = []
    factor_a_levels: set[str] = set()
    factor_b_levels: set[str] = set()
    seen_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    for index, columns in enumerate(_parse_rows(raw_value), start=1):
        if len(columns) != 3:
            issues.append(build_error(field, f"Line {index} must contain factor A, factor B, and value."))
            continue
        factor_a, factor_b, value_token = columns
        if not factor_a or not factor_b:
            issues.append(build_error(field, f"Line {index} must include both factor labels."))
            continue
        try:
            parsed_value = float(value_token)
        except ValueError:
            issues.append(build_error(field, f"Line {index} contains a non-numeric value: {value_token}."))
            continue

        factor_a_levels.add(factor_a)
        factor_b_levels.add(factor_b)
        seen_counts[(factor_a, factor_b)] += 1
        rows.append(TwoWayAnovaRow(factor_a=factor_a, factor_b=factor_b, value=parsed_value))

    if len(factor_a_levels) < 2:
        issues.append(build_error(field, "Provide at least two levels for factor A."))
    if len(factor_b_levels) < 2:
        issues.append(build_error(field, "Provide at least two levels for factor B."))
    if rows and any(count < 2 for count in seen_counts.values()):
        issues.append(build_error(field, "Provide at least two observations for every factor combination."))

    return tuple(rows) if not issues else None, issues



def parse_variable_names(value: object, field: str) -> tuple[tuple[str, ...] | None, list[ValidationIssue]]:
    raw_value = "" if value is None else str(value).strip()
    if not raw_value:
        return None, [build_error(field, "Response variable names are required.")]
    names = tuple(token.strip() for token in raw_value.split(",") if token.strip())
    if len(names) < 2:
        return None, [build_error(field, "Provide at least two response variable names.")]
    if len(set(names)) != len(names):
        return None, [build_error(field, "Response variable names must be unique.")]
    return names, []



def parse_manova_rows(
    rows_value: object,
    variable_names_value: object,
    *,
    rows_field: str = "rows",
    variable_names_field: str = "variable_names",
) -> tuple[tuple[str, ...] | None, tuple[ManovaRow, ...] | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    variable_names, variable_name_issues = parse_variable_names(variable_names_value, variable_names_field)
    issues.extend(variable_name_issues)

    raw_value = "" if rows_value is None else str(rows_value).strip()
    if not raw_value:
        issues.append(build_error(rows_field, "MANOVA rows are required."))
        return variable_names, None, issues

    rows: list[ManovaRow] = []
    group_counts: defaultdict[str, int] = defaultdict(int)
    expected_width = 1 + (len(variable_names) if variable_names is not None else 0)

    for index, columns in enumerate(_parse_rows(raw_value), start=1):
        if variable_names is not None and len(columns) != expected_width:
            issues.append(
                build_error(
                    rows_field,
                    f"Line {index} must contain one group label and {len(variable_names)} numeric responses.",
                )
            )
            continue
        if not columns or not columns[0]:
            issues.append(build_error(rows_field, f"Line {index} must include a group label."))
            continue
        group_name = columns[0]
        values: list[float] = []
        has_numeric_error = False
        for token in columns[1:]:
            try:
                values.append(float(token))
            except ValueError:
                has_numeric_error = True
                issues.append(build_error(rows_field, f"Line {index} contains a non-numeric value: {token}."))
        if has_numeric_error:
            continue
        group_counts[group_name] += 1
        rows.append(ManovaRow(group=group_name, values=tuple(values)))

    if len(group_counts) < 2:
        issues.append(build_error(rows_field, "Provide at least two groups for MANOVA."))
    if rows and any(count < 2 for count in group_counts.values()):
        issues.append(build_error(rows_field, "Each MANOVA group must contain at least two observations."))

    return variable_names, tuple(rows) if not issues else None, issues



def parse_count_trial_inputs(
    successes_value: object,
    trials_value: object,
    *,
    successes_field: str,
    trials_field: str,
    label_prefix: str,
) -> tuple[int | None, int | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    successes = parse_int(successes_value, successes_field, f"{label_prefix} successes", issues)
    trials = parse_int(trials_value, trials_field, f"{label_prefix} trials", issues)
    if successes is not None and successes < 0:
        issues.append(build_error(successes_field, f"{label_prefix} successes cannot be negative."))
    if trials is not None and trials <= 0:
        issues.append(build_error(trials_field, f"{label_prefix} trials must be greater than zero."))
    if successes is not None and trials is not None and successes > trials:
        issues.append(build_error(successes_field, f"{label_prefix} successes cannot exceed trials."))
    return successes, trials, issues



def parse_observed_expected(
    observed_value: object,
    expected_value: object,
    *,
    observed_field: str = "observed",
    expected_field: str = "expected",
) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    observed, observed_issues = parse_numeric_series(
        observed_value,
        observed_field,
        "Observed counts",
        minimum_length=2,
        nonnegative=True,
    )
    expected, expected_issues = parse_numeric_series(
        expected_value,
        expected_field,
        "Expected values",
        minimum_length=2,
        positive_only=True,
    )
    issues.extend(observed_issues)
    issues.extend(expected_issues)

    if observed is not None and expected is not None:
        if len(observed) != len(expected):
            issues.append(build_error(expected_field, "Observed counts and expected values must have the same length."))
        elif sum(observed) <= 0:
            issues.append(build_error(observed_field, "Observed counts must sum to a positive total."))
        elif isclose(sum(expected), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            expected = tuple(probability * sum(observed) for probability in expected)
        elif not isclose(sum(expected), sum(observed), rel_tol=1e-9, abs_tol=1e-9):
            issues.append(build_error(expected_field, "Expected counts must either sum to 1 as probabilities or match the observed total."))

    return observed, expected, issues



def parse_survival_rows(value: object, field: str) -> tuple[tuple[SurvivalObservation, ...] | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    raw_value = "" if value is None else str(value).strip()
    if not raw_value:
        issues.append(build_error(field, "Survival rows are required."))
        return None, issues

    rows: list[SurvivalObservation] = []
    for index, columns in enumerate(_parse_rows(raw_value), start=1):
        if len(columns) != 2:
            issues.append(build_error(field, f"Line {index} must contain time and event."))
            continue
        time_token, event_token = columns
        try:
            time_value = float(time_token)
        except ValueError:
            issues.append(build_error(field, f"Line {index} contains a non-numeric duration: {time_token}."))
            continue
        if time_value <= 0:
            issues.append(build_error(field, f"Line {index} must use a positive duration."))
            continue
        if event_token not in {"0", "1"}:
            issues.append(build_error(field, f"Line {index} must use 0 or 1 for the event flag."))
            continue
        rows.append(SurvivalObservation(time=time_value, event=int(event_token)))

    if len(rows) < 2:
        issues.append(build_error(field, "Provide at least two survival observations."))

    return tuple(rows) if not issues else None, issues



def parse_roc_rows(value: object, field: str, label: str) -> tuple[tuple[RocObservation, ...] | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    raw_value = "" if value is None else str(value).strip()
    if not raw_value:
        issues.append(build_error(field, f"{label} rows are required."))
        return None, issues

    rows: list[RocObservation] = []
    labels: set[int] = set()
    for index, columns in enumerate(_parse_rows(raw_value), start=1):
        if len(columns) != 2:
            issues.append(build_error(field, f"Line {index} must contain label and score."))
            continue
        label_token, score_token = columns
        if label_token not in {"0", "1"}:
            issues.append(build_error(field, f"Line {index} must use 0 or 1 as the binary label."))
            continue
        try:
            score = float(score_token)
        except ValueError:
            issues.append(build_error(field, f"Line {index} contains a non-numeric score: {score_token}."))
            continue
        parsed_label = int(label_token)
        labels.add(parsed_label)
        rows.append(RocObservation(label=parsed_label, score=score))

    if len(rows) < 3:
        issues.append(build_error(field, f"{label} must contain at least three rows."))
    if labels != {0, 1}:
        issues.append(build_error(field, f"{label} must include both positive and negative labels."))

    return tuple(rows) if not issues else None, issues



def parse_paired_roc_rows(value: object, field: str) -> tuple[tuple[PairedRocObservation, ...] | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    raw_value = "" if value is None else str(value).strip()
    if not raw_value:
        issues.append(build_error(field, "Paired ROC rows are required."))
        return None, issues

    rows: list[PairedRocObservation] = []
    labels: set[int] = set()
    for index, columns in enumerate(_parse_rows(raw_value), start=1):
        if len(columns) != 3:
            issues.append(build_error(field, f"Line {index} must contain label, score A, and score B."))
            continue
        label_token, score_a_token, score_b_token = columns
        if label_token not in {"0", "1"}:
            issues.append(build_error(field, f"Line {index} must use 0 or 1 as the binary label."))
            continue
        try:
            score_a = float(score_a_token)
            score_b = float(score_b_token)
        except ValueError:
            issues.append(build_error(field, f"Line {index} must contain numeric ROC scores."))
            continue
        parsed_label = int(label_token)
        labels.add(parsed_label)
        rows.append(PairedRocObservation(label=parsed_label, score_a=score_a, score_b=score_b))

    if len(rows) < 3:
        issues.append(build_error(field, "Paired ROC rows must contain at least three observations."))
    if labels != {0, 1}:
        issues.append(build_error(field, "Paired ROC rows must include both positive and negative labels."))

    return tuple(rows) if not issues else None, issues



def parse_ks_distribution(
    distribution_value: object,
    parameter_value: object,
    *,
    distribution_field: str = "distribution",
    parameter_field: str = "distribution_parameters",
) -> tuple[str | None, tuple[float, ...] | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    distribution = "" if distribution_value is None else str(distribution_value).strip()
    if distribution not in KS_DISTRIBUTIONS:
        issues.append(build_error(distribution_field, "Choose a valid reference distribution."))
        return None, None, issues

    raw_parameters = "" if parameter_value is None else str(parameter_value).strip()
    parameters: tuple[float, ...]
    if not raw_parameters:
        defaults = {
            "norm": (0.0, 1.0),
            "expon": (0.0, 1.0),
            "uniform": (0.0, 1.0),
        }
        parameters = defaults[distribution]
        return distribution, parameters, issues

    tokens = [token.strip() for token in raw_parameters.split(",") if token.strip()]
    try:
        parsed = tuple(float(token) for token in tokens)
    except ValueError:
        issues.append(build_error(parameter_field, "Distribution parameters must be comma-separated numbers."))
        return distribution, None, issues

    expected_lengths = {"norm": 2, "expon": 2, "uniform": 2}
    if len(parsed) != expected_lengths[distribution]:
        issues.append(build_error(parameter_field, "Provide exactly two distribution parameters."))
        return distribution, None, issues

    if parsed[1] <= 0:
        issues.append(build_error(parameter_field, "The scale parameter must be greater than zero."))
        return distribution, None, issues

    return distribution, parsed, issues
