from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import AnovaRM

from test_statistics.domain.enums import TestFamily
from test_statistics.domain.inputs import NamedGroupsInput, OneWayManovaInput, RepeatedMeasuresInput, TwoWayAnovaInput
from test_statistics.domain.metadata import CalculatorMetadata
from test_statistics.domain.results import CalculationResult, DecisionSummary, ResultMetric, ResultSection, ResultTable, display_number, display_p_value, format_number
from test_statistics.services.calculators.base import MultivariateCalculator, NamedGroupCalculator, alpha_field, textarea_field, text_field
from test_statistics.services.validators import parse_alpha, parse_manova_rows, parse_named_groups, parse_repeated_measures_rows, parse_two_way_rows, raise_if_issues


class OneWayAnovaInput(NamedGroupsInput):
    pass



def _anova_decision(test_name: str, reject_null: bool) -> str:
    if reject_null:
        return f"Reject the null hypothesis for the {test_name}."
    return f"Fail to reject the null hypothesis for the {test_name}."


class OneWayAnovaCalculator(NamedGroupCalculator):
    metadata = CalculatorMetadata(
        catalog_position=9,
        slug="one-way-anova",
        name="One Way ANOVA Test",
        family=TestFamily.ANOVA,
        description="Compare the means of three or more independent groups.",
        check="Whether at least one group mean differs from the others.",
        statistic_formula="F = MS_between / MS_within",
        assumptions=(
            "Observations are independent within and across groups.",
            "The response is approximately normally distributed within each group.",
            "Group variances are reasonably similar.",
        ),
        required_sample_data=(
            "At least two named groups of numeric values.",
            "Each group should contain at least two observations.",
        ),
        input_fields=(
            textarea_field("groups", "Group samples", "Enter one group per line, such as 'Control: 4, 5, 6'.", placeholder="Control: 4, 5, 6\nTreatment A: 6, 7, 8\nTreatment B: 8, 9, 10", rows=7),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> OneWayAnovaInput:
        issues = []
        groups, group_issues = parse_named_groups(raw_data.get("groups"), "groups")
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(group_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return OneWayAnovaInput(groups=groups, alpha=alpha)

    def calculate_result(self, normalized_input: OneWayAnovaInput) -> CalculationResult:
        groups = normalized_input.groups
        arrays = [np.asarray(group_values, dtype=float) for _, group_values in groups]
        result = stats.f_oneway(*arrays)
        group_names = [group_name for group_name, _ in groups]
        group_means = [float(values.mean()) for values in arrays]
        total_count = sum(values.size for values in arrays)
        group_count = len(arrays)
        grand_mean = float(np.concatenate(arrays).mean())
        ss_between = sum(values.size * (group_mean - grand_mean) ** 2 for values, group_mean in zip(arrays, group_means, strict=True))
        ss_total = sum(((value - grand_mean) ** 2).sum() for value in arrays)
        eta_squared = float(ss_between / ss_total) if ss_total else 0.0
        df_between = group_count - 1
        df_within = total_count - group_count
        reject_null = float(result.pvalue) < normalized_input.alpha
        metrics = (
            ResultMetric("Groups", str(group_count)),
            ResultMetric("Total observations", str(total_count)),
            ResultMetric("Between-groups df", str(df_between)),
            ResultMetric("Within-groups df", str(df_within)),
            ResultMetric(
                "Group means",
                "; ".join(f"{name}: {format_number(mean)}" for name, mean in zip(group_names, group_means, strict=True)),
                emphasis=True,
            ),
            ResultMetric("Eta squared", format_number(eta_squared)),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="F",
            statistic=display_number(float(result.statistic)),
            p_value=display_p_value(float(result.pvalue)),
            metrics=metrics,
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_anova_decision(self.metadata.name, reject_null)),
            interpretation=(
                f"Compared {group_count} groups with {total_count} total observations around a grand mean of {format_number(grand_mean)}."
            ),
        )


class RepeatedMeasuresAnovaCalculator(NamedGroupCalculator):
    metadata = CalculatorMetadata(
        catalog_position=10,
        slug="repeated-measures-anova",
        name="Repeated Measures ANOVA Test",
        family=TestFamily.ANOVA,
        description="Compare repeated measurements across conditions for the same subjects.",
        check="Whether at least one repeated-measures condition mean differs from the others.",
        statistic_formula="F = MS_condition / MS_error",
        assumptions=(
            "The same subjects are measured under each condition.",
            "Subjects are independent of each other.",
            "The residuals are approximately normal and the within-subject structure is appropriate.",
        ),
        required_sample_data=(
            "Repeated-measures rows in 'subject, condition, value' format.",
            "Each subject must appear once for every condition.",
        ),
        input_fields=(
            textarea_field("rows", "Repeated-measures rows", "Enter one observation per line as 'subject, condition, value'.", placeholder="S1, Baseline, 4\nS1, Treatment, 6\nS2, Baseline, 5\nS2, Treatment, 7", rows=8),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> RepeatedMeasuresInput:
        issues = []
        rows, row_issues = parse_repeated_measures_rows(raw_data.get("rows"), "rows")
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(row_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return RepeatedMeasuresInput(rows=rows, alpha=alpha)

    def calculate_result(self, normalized_input: RepeatedMeasuresInput) -> CalculationResult:
        dataframe = pd.DataFrame(
            [
                {"subject": row.subject, "condition": row.condition, "value": row.value}
                for row in normalized_input.rows
            ]
        )
        fitted = AnovaRM(dataframe, depvar="value", subject="subject", within=["condition"]).fit()
        anova_table = fitted.anova_table.reset_index().rename(columns={"index": "source"})
        effect_row = anova_table.iloc[0]
        statistic = float(effect_row["F Value"])
        p_value = float(effect_row["Pr > F"])
        reject_null = p_value < normalized_input.alpha

        condition_means = dataframe.groupby("condition")["value"].mean().sort_index()
        section = ResultSection(
            title="Condition means",
            metrics=tuple(
                ResultMetric(condition, format_number(float(mean)), emphasis=True)
                for condition, mean in condition_means.items()
            ),
        )
        table = ResultTable(
            title="Repeated-measures ANOVA table",
            columns=("Source", "F", "Num DF", "Den DF", "p-value"),
            rows=tuple(
                (
                    str(row["source"]),
                    format_number(float(row["F Value"])),
                    format_number(float(row["Num DF"])),
                    format_number(float(row["Den DF"])),
                    format_number(float(row["Pr > F"])),
                )
                for _, row in anova_table.iterrows()
            ),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="F",
            statistic=display_number(statistic),
            p_value=display_p_value(p_value),
            metrics=(
                ResultMetric("Subjects", str(dataframe["subject"].nunique())),
                ResultMetric("Conditions", str(dataframe["condition"].nunique())),
            ),
            sections=(section,),
            tables=(table,),
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_anova_decision(self.metadata.name, reject_null)),
            interpretation="The repeated-measures ANOVA summarizes how the mean response changes across within-subject conditions.",
        )


class TwoWayAnovaCalculator(NamedGroupCalculator):
    metadata = CalculatorMetadata(
        catalog_position=13,
        slug="two-way-anova",
        name="Two Way ANOVA Test",
        family=TestFamily.ANOVA,
        description="Estimate the main effects of two categorical factors and their interaction on a numeric response.",
        check="Whether factor A, factor B, or their interaction explains differences in the response.",
        statistic_formula="F tests for factor A, factor B, and factor A × factor B",
        assumptions=(
            "Observations are independent.",
            "Residuals are approximately normal.",
            "Variances are reasonably similar across cells.",
        ),
        required_sample_data=(
            "Rows in 'factor_a, factor_b, value' format.",
            "At least two observations for every factor combination.",
        ),
        input_fields=(
            textarea_field("rows", "Factor rows", "Enter one observation per line as 'factor_a, factor_b, value'.", placeholder="Low, Control, 4\nLow, Control, 5\nLow, Treatment, 7\nHigh, Control, 6\nHigh, Treatment, 9", rows=9),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> TwoWayAnovaInput:
        issues = []
        rows, row_issues = parse_two_way_rows(raw_data.get("rows"), "rows")
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(row_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return TwoWayAnovaInput(rows=rows, alpha=alpha)

    def calculate_result(self, normalized_input: TwoWayAnovaInput) -> CalculationResult:
        dataframe = pd.DataFrame(
            [
                {"factor_a": row.factor_a, "factor_b": row.factor_b, "value": row.value}
                for row in normalized_input.rows
            ]
        )
        model = ols("value ~ C(factor_a) * C(factor_b)", data=dataframe).fit()
        anova_table = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "source"})
        interaction_row = anova_table.loc[anova_table["source"] == "C(factor_a):C(factor_b)"].iloc[0]
        statistic = float(interaction_row["F"])
        p_value = float(interaction_row["PR(>F)"])
        reject_null = p_value < normalized_input.alpha
        cell_means = dataframe.groupby(["factor_a", "factor_b"])["value"].mean().reset_index()
        mean_table = ResultTable(
            title="Cell means",
            columns=("Factor A", "Factor B", "Mean"),
            rows=tuple(
                (
                    str(row["factor_a"]),
                    str(row["factor_b"]),
                    format_number(float(row["value"])),
                )
                for _, row in cell_means.iterrows()
            ),
        )
        anova_result_table = ResultTable(
            title="Two-way ANOVA table",
            columns=("Source", "Sum Sq", "DF", "F", "p-value"),
            rows=tuple(
                (
                    str(row["source"]),
                    format_number(float(row["sum_sq"])),
                    format_number(float(row["df"])),
                    format_number(float(row["F"])) if pd.notna(row["F"]) else "—",
                    format_number(float(row["PR(>F)"])) if pd.notna(row["PR(>F)"]) else "—",
                )
                for _, row in anova_table.iterrows()
            ),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="Interaction F",
            statistic=display_number(statistic),
            p_value=display_p_value(p_value),
            metrics=(
                ResultMetric("Factor A levels", str(dataframe["factor_a"].nunique())),
                ResultMetric("Factor B levels", str(dataframe["factor_b"].nunique())),
                ResultMetric("Observations", str(len(dataframe))),
            ),
            tables=(anova_result_table, mean_table),
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_anova_decision(self.metadata.name, reject_null)),
            interpretation="The primary hero metric reports the interaction term; the ANOVA table includes both main effects as well.",
            notes=("Inspect the full ANOVA table for the separate factor A and factor B tests.",),
        )


class OneWayManovaCalculator(MultivariateCalculator):
    metadata = CalculatorMetadata(
        catalog_position=14,
        slug="one-way-manova",
        name="One Way MANOVA Test",
        family=TestFamily.MULTIVARIATE,
        description="Compare multivariate mean vectors across independent groups.",
        check="Whether at least one group's multivariate response profile differs from the others.",
        statistic_formula="Multivariate omnibus tests such as Pillai's trace, Wilks' lambda, Hotelling-Lawley trace, and Roy's greatest root",
        assumptions=(
            "Observations are independent.",
            "The multivariate response is approximately normal within groups.",
            "Group covariance matrices are reasonably similar.",
        ),
        required_sample_data=(
            "Comma-separated response variable names.",
            "Rows in 'group, value1, value2, ...' format.",
        ),
        input_fields=(
            text_field("variable_names", "Response variable names", "Enter comma-separated response variable names in the same order as each row.", placeholder="score_1, score_2"),
            textarea_field("rows", "MANOVA rows", "Enter one observation per line as 'group, value1, value2, ...'.", placeholder="Control, 10, 15\nControl, 11, 14\nTreatment, 15, 18\nTreatment, 16, 19", rows=9),
            alpha_field(),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> OneWayManovaInput:
        variable_names, rows, issues = parse_manova_rows(raw_data.get("rows"), raw_data.get("variable_names"))
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return OneWayManovaInput(variable_names=variable_names, rows=rows, alpha=alpha)

    def calculate_result(self, normalized_input: OneWayManovaInput) -> CalculationResult:
        records = []
        for row in normalized_input.rows:
            record = {"group": row.group}
            for variable_name, value in zip(normalized_input.variable_names, row.values, strict=True):
                record[variable_name] = value
            records.append(record)
        dataframe = pd.DataFrame(records)
        response_formula = " + ".join(normalized_input.variable_names)
        fitted = MANOVA.from_formula(f"{response_formula} ~ group", data=dataframe).mv_test()
        stat_table = fitted.results["group"]["stat"].reset_index().rename(columns={"index": "Statistic"})
        pillai_row = stat_table.loc[stat_table["Statistic"] == "Pillai's trace"].iloc[0]
        statistic = float(pillai_row["Value"])
        p_value = float(pillai_row["Pr > F"])
        reject_null = p_value < normalized_input.alpha
        group_means = dataframe.groupby("group").mean(numeric_only=True)
        mean_sections = tuple(
            ResultSection(
                title=f"{group_name} means",
                metrics=tuple(
                    ResultMetric(variable_name, format_number(float(mean)), emphasis=True)
                    for variable_name, mean in values.items()
                ),
            )
            for group_name, values in group_means.iterrows()
        )
        stat_result_table = ResultTable(
            title="MANOVA multivariate tests",
            columns=("Statistic", "Value", "Num DF", "Den DF", "F", "p-value"),
            rows=tuple(
                (
                    str(row["Statistic"]),
                    format_number(float(row["Value"])),
                    format_number(float(row["Num DF"])),
                    format_number(float(row["Den DF"])),
                    format_number(float(row["F Value"])),
                    format_number(float(row["Pr > F"])),
                )
                for _, row in stat_table.iterrows()
            ),
        )
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="Pillai's trace",
            statistic=display_number(statistic),
            p_value=display_p_value(p_value),
            metrics=(
                ResultMetric("Groups", str(dataframe["group"].nunique())),
                ResultMetric("Observations", str(len(dataframe))),
                ResultMetric("Response variables", str(len(normalized_input.variable_names))),
            ),
            sections=mean_sections,
            tables=(stat_result_table,),
            decision=DecisionSummary(alpha=normalized_input.alpha, reject_null=reject_null, conclusion=_anova_decision(self.metadata.name, reject_null)),
            interpretation="Pillai's trace is reported as the hero statistic because it is a stable omnibus multivariate test.",
            notes=("Inspect the MANOVA table for Wilks' lambda, Hotelling-Lawley trace, and Roy's greatest root alongside Pillai's trace.",),
        )
