from django.test import SimpleTestCase

from services.validators import (
    ValidationIssues,
    _parse_rows,
    build_error,
    errors_by_field,
    parse_alpha,
    parse_alternative,
    parse_count_trial_inputs,
    parse_float,
    parse_int,
    parse_ks_distribution,
    parse_manova_rows,
    parse_named_groups,
    parse_numeric_series,
    parse_observed_expected,
    parse_paired_roc_rows,
    parse_probability,
    parse_positive_float,
    parse_repeated_measures_rows,
    parse_roc_rows,
    parse_survival_rows,
    parse_two_way_rows,
    parse_variable_names,
    raise_if_issues,
)


class ValidatorHelperTests(SimpleTestCase):
    def test_parse_float_handles_blank_and_non_numeric(self):
        issues = []
        self.assertIsNone(parse_float("", "value", "Value", issues))
        self.assertIsNone(parse_float("abc", "value", "Value", issues))
        self.assertEqual(errors_by_field(issues)["value"], ["Value is required.", "Value must be a number."])

    def test_parse_int_and_probability_bounds(self):
        issues = []
        self.assertIsNone(parse_int("1.2", "count", "Count", issues))
        self.assertIsNone(parse_probability("2", "p", "Probability", issues))
        self.assertEqual(
            errors_by_field(issues),
            {"count": ["Count must be an integer."], "p": ["Probability must be between 0 and 1."]},
        )

    def test_parse_positive_float_zero_is_invalid(self):
        issues = []
        self.assertIsNone(parse_positive_float("0", "rate", "Rate", issues))
        self.assertEqual(errors_by_field(issues), {"rate": ["Rate must be greater than zero."]})

    def test_parse_alpha_and_alternative_invalid_values(self):
        alpha, alpha_issues = parse_alpha("1")
        alternative, alt_issues = parse_alternative("invalid")
        self.assertEqual(alpha, 1.0)
        self.assertIsNone(alternative)
        self.assertEqual(errors_by_field(alpha_issues), {"alpha": ["Alpha must be between 0 and 1."]})
        self.assertEqual(errors_by_field(alt_issues), {"alternative": ["Choose a valid alternative hypothesis."]})

    def test_parse_numeric_series_handles_non_numeric_and_minimum_length(self):
        values, issues = parse_numeric_series("1, bad", "series", "Series", minimum_length=3)
        self.assertIsNone(values)
        self.assertEqual(
            errors_by_field(issues),
            {
                "series": [
                    "Series contains a non-numeric value: bad.",
                    "Series must contain at least 3 numeric values.",
                ]
            },
        )

    def test_parse_numeric_series_nonnegative_and_positive_constraints(self):
        nonnegative_values, nonnegative_issues = parse_numeric_series(
            "1,-1", "observed", "Observed", nonnegative=True
        )
        positive_values, positive_issues = parse_numeric_series("1,0", "expected", "Expected", positive_only=True)

        self.assertIsNone(nonnegative_values)
        self.assertIsNone(positive_values)
        self.assertEqual(
            errors_by_field(nonnegative_issues),
            {
                "observed": [
                    "Observed cannot include negative values.",
                    "Observed must contain at least 2 numeric values.",
                ]
            },
        )
        self.assertEqual(
            errors_by_field(positive_issues),
            {
                "expected": [
                    "Expected must contain only positive values.",
                    "Expected must contain at least 2 numeric values.",
                ]
            },
        )

    def test_parse_count_trial_inputs_rejects_negative_and_exceeding(self):
        successes, trials, issues = parse_count_trial_inputs(
            -1,
            0,
            successes_field="s",
            trials_field="t",
            label_prefix="Sample",
        )
        self.assertEqual((successes, trials), (-1, 0))
        self.assertEqual(
            errors_by_field(issues),
            {"s": ["Sample successes cannot be negative."], "t": ["Sample trials must be greater than zero."]},
        )

        _, _, issues_exceeding = parse_count_trial_inputs(
            5,
            4,
            successes_field="s",
            trials_field="t",
            label_prefix="Sample",
        )
        self.assertEqual(errors_by_field(issues_exceeding), {"s": ["Sample successes cannot exceed trials."]})

    def test_parse_observed_expected_probability_conversion_and_sum_validation(self):
        observed, expected, issues = parse_observed_expected("10, 20", "0.25, 0.75")
        self.assertEqual(observed, (10.0, 20.0))
        self.assertEqual(expected, (7.5, 22.5))
        self.assertEqual(issues, [])

        _, _, sum_issues = parse_observed_expected("10, 20", "10, 10")
        self.assertEqual(
            errors_by_field(sum_issues),
            {
                "expected": [
                    "Expected counts must either sum to 1 as probabilities or match the observed total."
                ]
            },
        )

    def test_parse_ks_distribution_validation(self):
        dist, params, issues = parse_ks_distribution("bad", "")
        self.assertIsNone(dist)
        self.assertIsNone(params)
        self.assertEqual(errors_by_field(issues), {"distribution": ["Choose a valid reference distribution."]})

        dist, params, issues = parse_ks_distribution("norm", "1,2,3")
        self.assertEqual(dist, "norm")
        self.assertIsNone(params)
        self.assertEqual(errors_by_field(issues), {"distribution_parameters": ["Provide exactly two distribution parameters."]})

    def test_parse_int_blank_is_required(self):
        issues = []
        self.assertIsNone(parse_int("", "count", "Count", issues))
        self.assertEqual(errors_by_field(issues), {"count": ["Count is required."]})

    def test_parse_numeric_series_blank_and_only_separators(self):
        values_blank, issues_blank = parse_numeric_series("", "series", "Series")
        values_separators, issues_separators = parse_numeric_series(",\n,", "series", "Series", minimum_length=3)

        self.assertIsNone(values_blank)
        self.assertIsNone(values_separators)
        self.assertEqual(errors_by_field(issues_blank), {"series": ["Series is required."]})
        self.assertEqual(errors_by_field(issues_separators), {"series": ["Series must contain at least 3 numeric values."]})

    def test_parse_rows_skips_blank_lines(self):
        rows = _parse_rows("A,1\n\nB,2")
        self.assertEqual(rows, [["A", "1"], ["B", "2"]])

    def test_parse_named_groups_reports_structural_errors(self):
        groups, issues = parse_named_groups(
            "GroupA 1,2,3\n: 1,2\nA:1,2\nA:3,4",
            "groups",
            minimum_groups=2,
            minimum_length=2,
        )

        self.assertIsNone(groups)
        self.assertEqual(
            errors_by_field(issues),
            {
                "groups": [
                    "Line 1 must look like 'Group A: 1, 2, 3'.",
                    "Line 2 must include a group name.",
                    "Group name 'A' is duplicated.",
                    "Provide at least 2 groups.",
                ]
            },
        )

    def test_parse_named_groups_requires_input(self):
        groups, issues = parse_named_groups("", "groups")
        self.assertIsNone(groups)
        self.assertEqual(errors_by_field(issues), {"groups": ["Group samples are required."]})

    def test_parse_repeated_measures_rows_error_paths(self):
        rows, issues = parse_repeated_measures_rows("S1,Baseline\nS1,Baseline,5\nS1,,5\nS1,Final,bad\nS1,Baseline,6", "rows")

        self.assertIsNone(rows)
        row_errors = errors_by_field(issues)["rows"]
        self.assertIn("Line 1 must contain subject, condition, and value.", row_errors)
        self.assertTrue(any("must include both a subject and a condition" in message for message in row_errors))
        self.assertTrue(any("contains a non-numeric value: bad." in message for message in row_errors))
        self.assertIn("The subject/condition pair 'S1, Baseline' is duplicated.", row_errors)
        self.assertIn("Provide at least two subjects.", row_errors)
        self.assertIn("Provide at least two conditions.", row_errors)

    def test_parse_repeated_measures_rows_requires_input(self):
        rows, issues = parse_repeated_measures_rows("", "rows")
        self.assertIsNone(rows)
        self.assertEqual(errors_by_field(issues), {"rows": ["Repeated-measures rows are required."]})

    def test_parse_two_way_rows_error_paths(self):
        rows, issues = parse_two_way_rows("A1,B1\nA1,,1\nA1,B1,bad\nA1,B1,2", "rows")

        self.assertIsNone(rows)
        self.assertEqual(
            errors_by_field(issues),
            {
                "rows": [
                    "Line 1 must contain factor A, factor B, and value.",
                    "Line 2 must include both factor labels.",
                    "Line 3 contains a non-numeric value: bad.",
                    "Provide at least two levels for factor A.",
                    "Provide at least two levels for factor B.",
                    "Provide at least two observations for every factor combination.",
                ]
            },
        )

    def test_parse_two_way_rows_requires_input(self):
        rows, issues = parse_two_way_rows("", "rows")
        self.assertIsNone(rows)
        self.assertEqual(errors_by_field(issues), {"rows": ["Two-way ANOVA rows are required."]})

    def test_parse_variable_names_validation_errors(self):
        missing, missing_issues = parse_variable_names("", "variable_names")
        too_few, too_few_issues = parse_variable_names("x", "variable_names")
        duplicate, duplicate_issues = parse_variable_names("x, x", "variable_names")

        self.assertIsNone(missing)
        self.assertIsNone(too_few)
        self.assertIsNone(duplicate)
        self.assertEqual(errors_by_field(missing_issues), {"variable_names": ["Response variable names are required."]})
        self.assertEqual(errors_by_field(too_few_issues), {"variable_names": ["Provide at least two response variable names."]})
        self.assertEqual(errors_by_field(duplicate_issues), {"variable_names": ["Response variable names must be unique."]})

    def test_parse_manova_rows_error_paths(self):
        variable_names, rows, issues = parse_manova_rows(
            "\n,1,2\nG1,1,bad\nG1,1,2\n",
            "x,y",
            rows_field="rows",
            variable_names_field="variable_names",
        )

        self.assertEqual(variable_names, ("x", "y"))
        self.assertIsNone(rows)
        self.assertEqual(
            errors_by_field(issues),
            {
                "rows": [
                    "Line 1 must include a group label.",
                    "Line 2 contains a non-numeric value: bad.",
                    "Provide at least two groups for MANOVA.",
                    "Each MANOVA group must contain at least two observations.",
                ]
            },
        )

    def test_parse_manova_rows_requires_rows(self):
        variable_names, rows, issues = parse_manova_rows("", "x,y")
        self.assertEqual(variable_names, ("x", "y"))
        self.assertIsNone(rows)
        self.assertEqual(errors_by_field(issues), {"rows": ["MANOVA rows are required."]})

    def test_parse_manova_rows_wrong_width(self):
        _, rows, issues = parse_manova_rows("G1,1\nG2,2,3,4", "x,y")
        self.assertIsNone(rows)
        self.assertEqual(
            errors_by_field(issues),
            {
                "rows": [
                    "Line 1 must contain one group label and 2 numeric responses.",
                    "Line 2 must contain one group label and 2 numeric responses.",
                    "Provide at least two groups for MANOVA.",
                ]
            },
        )

    def test_parse_observed_expected_rejects_non_positive_observed_sum(self):
        observed, expected, issues = parse_observed_expected("0,0", "0.5,0.5")
        self.assertEqual(observed, (0.0, 0.0))
        self.assertEqual(expected, (0.5, 0.5))
        self.assertEqual(errors_by_field(issues), {"observed": ["Observed counts must sum to a positive total."]})

    def test_parse_survival_rows_error_paths(self):
        rows, issues = parse_survival_rows("1\nabc,1\n0,1\n2,2", "rows")
        self.assertIsNone(rows)
        self.assertEqual(
            errors_by_field(issues),
            {
                "rows": [
                    "Line 1 must contain time and event.",
                    "Line 2 contains a non-numeric duration: abc.",
                    "Line 3 must use a positive duration.",
                    "Line 4 must use 0 or 1 for the event flag.",
                    "Provide at least two survival observations.",
                ]
            },
        )

    def test_parse_survival_rows_requires_input(self):
        rows, issues = parse_survival_rows("", "rows")
        self.assertIsNone(rows)
        self.assertEqual(errors_by_field(issues), {"rows": ["Survival rows are required."]})

    def test_parse_roc_rows_error_paths(self):
        rows, issues = parse_roc_rows("1\n2,0.4\n1,bad\n1,0.8", "rows", "ROC")
        self.assertIsNone(rows)
        self.assertEqual(
            errors_by_field(issues),
            {
                "rows": [
                    "Line 1 must contain label and score.",
                    "Line 2 must use 0 or 1 as the binary label.",
                    "Line 3 contains a non-numeric score: bad.",
                    "ROC must contain at least three rows.",
                    "ROC must include both positive and negative labels.",
                ]
            },
        )

    def test_parse_roc_rows_requires_input(self):
        rows, issues = parse_roc_rows("", "rows", "ROC")
        self.assertIsNone(rows)
        self.assertEqual(errors_by_field(issues), {"rows": ["ROC rows are required."]})

    def test_parse_paired_roc_rows_error_paths(self):
        rows, issues = parse_paired_roc_rows("1,0.2\n2,0.3,0.4\n1,bad,0.5\n1,0.6,0.7", "rows")
        self.assertIsNone(rows)
        self.assertEqual(
            errors_by_field(issues),
            {
                "rows": [
                    "Line 1 must contain label, score A, and score B.",
                    "Line 2 must use 0 or 1 as the binary label.",
                    "Line 3 must contain numeric ROC scores.",
                    "Paired ROC rows must contain at least three observations.",
                    "Paired ROC rows must include both positive and negative labels.",
                ]
            },
        )

    def test_parse_paired_roc_rows_requires_input(self):
        rows, issues = parse_paired_roc_rows("", "rows")
        self.assertIsNone(rows)
        self.assertEqual(errors_by_field(issues), {"rows": ["Paired ROC rows are required."]})

    def test_parse_ks_distribution_default_and_non_numeric_parameters(self):
        dist, params, issues = parse_ks_distribution("uniform", "")
        self.assertEqual((dist, params, issues), ("uniform", (0.0, 1.0), []))

        dist, params, issues = parse_ks_distribution("norm", "0,abc")
        self.assertEqual(dist, "norm")
        self.assertIsNone(params)
        self.assertEqual(
            errors_by_field(issues),
            {"distribution_parameters": ["Distribution parameters must be comma-separated numbers."]},
        )

    def test_raise_if_issues_raises_validationissues(self):
        issue = build_error("field", "problem")
        with self.assertRaises(ValidationIssues):
            raise_if_issues([issue])
