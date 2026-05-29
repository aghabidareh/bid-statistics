from enum import Enum


class TestFamily(str, Enum):
    PARAMETRIC = "Parametric tests"
    NONPARAMETRIC = "Nonparametric tests"
    ANOVA = "ANOVA"
    MULTIVARIATE = "Multivariate analysis"
    PROPORTIONS = "Proportion tests"
    VARIANCE = "Variance tests"
    DISTRIBUTION = "Distribution tests"
    SURVIVAL = "Survival analysis"
    ROC = "ROC comparison"
    CORRELATION = "Correlation"
    REGRESSION = "Regression"
    CLASSIFICATION = "Classification"
    CAUSAL_INFERENCE = "Causal inference"


class InputKind(str, Enum):
    TEXT = "text"
    TEXTAREA = "textarea"
    NUMBER = "number"
    SELECT = "select"
    RADIO = "radio"


class WorkflowKind(str, Enum):
    FORM = "form"
    DATASET = "dataset"


class DatasetColumnRole(str, Enum):
    PREDICTOR = "predictor"
    TARGET = "target"
    TREATMENT = "treatment"
    OUTCOME = "outcome"
    ID = "id"
    UNUSED = "unused"


class AlternativeHypothesis(str, Enum):
    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"
