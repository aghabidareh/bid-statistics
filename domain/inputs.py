from __future__ import annotations

from dataclasses import dataclass

from domain.enums import AlternativeHypothesis


@dataclass(frozen=True)
class SingleSampleMeanInput:
    sample: tuple[float, ...]
    null_mean: float
    alternative: AlternativeHypothesis
    alpha: float


@dataclass(frozen=True)
class OneSampleZTestInput(SingleSampleMeanInput):
    known_std: float


@dataclass(frozen=True)
class OneSampleTTestInput(SingleSampleMeanInput):
    pass


@dataclass(frozen=True)
class TwoIndependentSampleMeanInput:
    sample_a: tuple[float, ...]
    sample_b: tuple[float, ...]
    alternative: AlternativeHypothesis
    alpha: float


@dataclass(frozen=True)
class TwoSampleZTestInput(TwoIndependentSampleMeanInput):
    known_std_a: float
    known_std_b: float


@dataclass(frozen=True)
class TwoSampleTTestInput(TwoIndependentSampleMeanInput):
    pass


@dataclass(frozen=True)
class WelchTTestInput(TwoIndependentSampleMeanInput):
    pass


@dataclass(frozen=True)
class MannWhitneyInput(TwoIndependentSampleMeanInput):
    pass


@dataclass(frozen=True)
class PairedSampleInput:
    sample_a: tuple[float, ...]
    sample_b: tuple[float, ...]
    alternative: AlternativeHypothesis
    alpha: float


@dataclass(frozen=True)
class PairedTTestInput(PairedSampleInput):
    pass


@dataclass(frozen=True)
class PairedWilcoxonInput(PairedSampleInput):
    pass


@dataclass(frozen=True)
class NamedGroupsInput:
    groups: tuple[tuple[str, tuple[float, ...]], ...]
    alpha: float


@dataclass(frozen=True)
class RepeatedMeasureRow:
    subject: str
    condition: str
    value: float


@dataclass(frozen=True)
class RepeatedMeasuresInput:
    rows: tuple[RepeatedMeasureRow, ...]
    alpha: float


@dataclass(frozen=True)
class TwoWayAnovaRow:
    factor_a: str
    factor_b: str
    value: float


@dataclass(frozen=True)
class TwoWayAnovaInput:
    rows: tuple[TwoWayAnovaRow, ...]
    alpha: float


@dataclass(frozen=True)
class ManovaRow:
    group: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class OneWayManovaInput:
    variable_names: tuple[str, ...]
    rows: tuple[ManovaRow, ...]
    alpha: float


@dataclass(frozen=True)
class OneSampleProportionInput:
    successes: int
    trials: int
    null_proportion: float
    alternative: AlternativeHypothesis
    alpha: float


@dataclass(frozen=True)
class TwoSampleProportionInput:
    successes_a: int
    trials_a: int
    successes_b: int
    trials_b: int
    alternative: AlternativeHypothesis
    alpha: float


@dataclass(frozen=True)
class OneSampleVarianceInput:
    sample: tuple[float, ...]
    null_variance: float
    alternative: AlternativeHypothesis
    alpha: float


@dataclass(frozen=True)
class TwoSampleVarianceInput:
    sample_a: tuple[float, ...]
    sample_b: tuple[float, ...]
    alternative: AlternativeHypothesis
    alpha: float


@dataclass(frozen=True)
class GoodnessOfFitInput:
    observed: tuple[float, ...]
    expected: tuple[float, ...]
    alpha: float


@dataclass(frozen=True)
class ShapiroWilkInput:
    sample: tuple[float, ...]
    alpha: float


@dataclass(frozen=True)
class OneSampleKsInput:
    sample: tuple[float, ...]
    distribution: str
    parameters: tuple[float, ...]
    alternative: AlternativeHypothesis
    alpha: float


@dataclass(frozen=True)
class TwoSampleKsInput:
    sample_a: tuple[float, ...]
    sample_b: tuple[float, ...]
    alternative: AlternativeHypothesis
    alpha: float


@dataclass(frozen=True)
class SurvivalObservation:
    time: float
    event: int


@dataclass(frozen=True)
class KaplanMeierInput:
    observations: tuple[SurvivalObservation, ...]
    alpha: float


@dataclass(frozen=True)
class RocObservation:
    label: int
    score: float


@dataclass(frozen=True)
class IndependentDelongInput:
    curve_a: tuple[RocObservation, ...]
    curve_b: tuple[RocObservation, ...]
    alpha: float


@dataclass(frozen=True)
class PairedRocObservation:
    label: int
    score_a: float
    score_b: float


@dataclass(frozen=True)
class PairedDelongInput:
    observations: tuple[PairedRocObservation, ...]
    alpha: float
