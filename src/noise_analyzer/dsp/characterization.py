from dataclasses import dataclass

from noise_analyzer.dsp.harmonics import (
    AutomaticHarmonicSearchResult,
)
from noise_analyzer.dsp.narrowband import (
    NarrowbandCharacterizationResult,
)
from noise_analyzer.dsp.spectral_features import (
    SpectralFlatnessResult,
)


@dataclass(frozen=True)
class CharacterizationResult:
    """Conservative characterization based on validated spectral evidence."""

    label: str
    spectral_flatness: float
    strong_line_count: int
    strongest_line_to_floor_db: float | None
    harmonic_match_count: int

    broadband_flatness_threshold: float
    tonal_flatness_threshold: float
    strong_line_threshold_db: float
    harmonic_minimum_matches: int


def characterize_spectral_behavior(
    spectral_flatness: SpectralFlatnessResult,
    narrowband_lines: NarrowbandCharacterizationResult,
    harmonic_family: AutomaticHarmonicSearchResult | None,
    broadband_flatness_threshold: float = 0.80,
    tonal_flatness_threshold: float = 0.10,
    strong_line_threshold_db: float = 10.0,
    harmonic_minimum_matches: int = 3,
) -> CharacterizationResult:
    """Combine validated measurements into a conservative spectral label."""

    if not 0.0 <= tonal_flatness_threshold <= 1.0:
        raise ValueError(
            "tonal_flatness_threshold must be between 0 and 1."
        )

    if not 0.0 <= broadband_flatness_threshold <= 1.0:
        raise ValueError(
            "broadband_flatness_threshold must be between 0 and 1."
        )

    if tonal_flatness_threshold >= broadband_flatness_threshold:
        raise ValueError(
            "tonal_flatness_threshold must be smaller than "
            "broadband_flatness_threshold."
        )

    if strong_line_threshold_db < 0:
        raise ValueError(
            "strong_line_threshold_db must be non-negative."
        )

    if harmonic_minimum_matches < 2:
        raise ValueError(
            "harmonic_minimum_matches must be at least 2."
        )

    flatness = spectral_flatness.spectral_flatness

    strong_lines = [
        line
        for line in narrowband_lines.lines
        if line.line_to_floor_db >= strong_line_threshold_db
    ]

    strong_line_count = len(strong_lines)

    if strong_lines:
        strongest_line_to_floor_db = max(
            line.line_to_floor_db
            for line in strong_lines
        )
    else:
        strongest_line_to_floor_db = None

    harmonic_match_count = 0

    if (
        harmonic_family is not None
        and harmonic_family.best_family is not None
    ):
        harmonic_match_count = len(
            harmonic_family.best_family.matches
        )

    if (
        harmonic_match_count >= harmonic_minimum_matches
        and strong_line_count >= harmonic_minimum_matches
    ):
        label = "harmonic-rich"

    elif (
        flatness <= tonal_flatness_threshold
        and strong_line_count >= 1
    ):
        label = "tonal-dominant"

    elif (
        flatness >= broadband_flatness_threshold
    ):
        label = "broadband-dominant"

    elif (
        strong_line_count >= 1
    ):
        label = "mixed"

    else:
        label = "undetermined"

    return CharacterizationResult(
        label=label,
        spectral_flatness=float(flatness),
        strong_line_count=strong_line_count,
        strongest_line_to_floor_db=strongest_line_to_floor_db,
        harmonic_match_count=harmonic_match_count,
        broadband_flatness_threshold=broadband_flatness_threshold,
        tonal_flatness_threshold=tonal_flatness_threshold,
        strong_line_threshold_db=strong_line_threshold_db,
        harmonic_minimum_matches=harmonic_minimum_matches,
    )