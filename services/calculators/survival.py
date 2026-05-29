from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from lifelines import KaplanMeierFitter

from domain.enums import TestFamily
from domain.inputs import KaplanMeierInput
from domain.metadata import CalculatorMetadata
from domain.results import CalculationResult, ResultMetric, ResultSection, ResultTable, display_number, format_number
from services.calculators.base import SurvivalCalculator, alpha_field, textarea_field
from services.validators import parse_alpha, parse_survival_rows, raise_if_issues


class KaplanMeierCalculator(SurvivalCalculator):
    metadata = CalculatorMetadata(
        catalog_position=24,
        slug="kaplan-meier-survival-analysis",
        name="Kaplan-Meier Survival Analysis",
        family=TestFamily.SURVIVAL,
        description="Estimate the survival function for one cohort with right-censored observations.",
        check="How survival probability changes over time in the presence of censoring.",
        statistic_formula="Ŝ(t) = Π (1 - dᵢ / nᵢ)",
        assumptions=(
            "Censoring is non-informative.",
            "Survival probabilities are similar for subjects recruited early and late.",
            "Event times are measured consistently.",
        ),
        required_sample_data=(
            "Rows in 'time, event' format.",
            "Use event = 1 for observed events and 0 for censored observations.",
        ),
        input_fields=(
            textarea_field("rows", "Survival rows", "Enter one observation per line as 'time, event'.", placeholder="5, 1\n8, 0\n12, 1\n15, 1", rows=8),
            alpha_field(help_text="Set alpha to control the displayed confidence interval level."),
        ),
    )

    def normalize(self, raw_data: Mapping[str, Any]) -> KaplanMeierInput:
        issues = []
        observations, observation_issues = parse_survival_rows(raw_data.get("rows"), "rows")
        alpha, alpha_issues = parse_alpha(raw_data.get("alpha"))
        issues.extend(observation_issues)
        issues.extend(alpha_issues)
        raise_if_issues(issues)
        return KaplanMeierInput(observations=observations, alpha=alpha)

    def calculate_result(self, normalized_input: KaplanMeierInput) -> CalculationResult:
        dataframe = pd.DataFrame(
            [{"time": observation.time, "event": observation.event} for observation in normalized_input.observations]
        )
        fitter = KaplanMeierFitter(alpha=normalized_input.alpha)
        fitter.fit(durations=dataframe["time"], event_observed=dataframe["event"], label="Overall survival")
        confidence_interval = fitter.confidence_interval_.reset_index().rename(columns={"index": "timeline"})
        survival_function = fitter.survival_function_.reset_index().rename(columns={"index": "timeline"})
        event_table = fitter.event_table.reset_index().rename(columns={"event_at": "timeline"})

        merged = survival_function.merge(confidence_interval, on="timeline", how="left")
        survival_column = fitter.survival_function_.columns[0]
        ci_columns = tuple(fitter.confidence_interval_.columns)
        median_survival = fitter.median_survival_time_
        statistic = None if pd.isna(median_survival) else float(median_survival)
        total = len(dataframe)
        events = int(dataframe["event"].sum())
        censored = total - events
        last_survival = float(fitter.survival_function_.iloc[-1, 0])

        survival_table = ResultTable(
            title="Survival function",
            columns=("Time", "Survival", "CI lower", "CI upper"),
            rows=tuple(
                (
                    format_number(float(row["timeline"])),
                    format_number(float(row[survival_column])),
                    format_number(float(row[ci_columns[0]])),
                    format_number(float(row[ci_columns[1]])),
                )
                for _, row in merged.iterrows()
            ),
        )
        event_summary_table = ResultTable(
            title="Event table",
            columns=("Time", "At risk", "Observed", "Censored", "Removed"),
            rows=tuple(
                (
                    format_number(float(row["timeline"])),
                    str(int(row["at_risk"])),
                    str(int(row["observed"])),
                    str(int(row["censored"])),
                    str(int(row["removed"])),
                )
                for _, row in event_table.iterrows()
            ),
        )
        sections = (
            ResultSection(
                title="Survival summary",
                metrics=(
                    ResultMetric("Observations", str(total), emphasis=True),
                    ResultMetric("Observed events", str(events), emphasis=True),
                    ResultMetric("Censored observations", str(censored)),
                    ResultMetric("Final survival estimate", format_number(last_survival)),
                ),
            ),
        )
        notes = ()
        if statistic is None:
            notes = ("Median survival was not reached within the observed follow-up window.",)
        return CalculationResult(
            slug=self.metadata.slug,
            test_name=self.metadata.name,
            statistic_name="Median survival time",
            statistic=display_number(statistic, empty="Not reached"),
            p_value=None,
            metrics=(
                ResultMetric("Confidence level", f"{int((1 - normalized_input.alpha) * 100)}%"),
            ),
            sections=sections,
            tables=(survival_table, event_summary_table),
            interpretation="Kaplan-Meier survival analysis estimates the probability of remaining event-free over time while accounting for censoring.",
            notes=notes,
        )
