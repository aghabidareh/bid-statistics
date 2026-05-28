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


class InputKind(str, Enum):
    TEXT = "text"
    TEXTAREA = "textarea"
    NUMBER = "number"
    SELECT = "select"
    RADIO = "radio"


class AlternativeHypothesis(str, Enum):
    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"
