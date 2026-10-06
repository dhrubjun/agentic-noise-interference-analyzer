from dataclasses import dataclass

import numpy as np

from noise_analyzer.dsp.peaks import PeakDetectionResult


@dataclass(frozen=True)
class HarmonicMatch:
    harmonic_order: int
    expected_frequency_hz: float
    detected_frequency_hz: float
    frequency_error_hz: float
    allowed_error_hz: float


@dataclass(frozen=True)
class HarmonicFamilyResult:
    candidate_fundamental_hz: float
    matches: tuple[HarmonicMatch, ...]
    tolerance_hz: float
    match_fraction: float
    longest_consecutive_run: int


def detect_harmonic_family(
    peaks: PeakDetectionResult,
    candidate_fundamental_hz: float,
    tolerance_hz: float,
    max_harmonic_order: int = 10,
    frequency_resolution_hz: float | None = None,
    relative_tolerance_fraction: float = 0.01,
) -> HarmonicFamilyResult:
    """Find peaks near integer multiples of a candidate fundamental."""

    if candidate_fundamental_hz <= 0:
        raise ValueError(
            "candidate_fundamental_hz must be greater than zero."
        )

    if tolerance_hz < 0:
        raise ValueError(
            "tolerance_hz must be non-negative."
        )

    if max_harmonic_order < 1:
        raise ValueError(
            "max_harmonic_order must be at least 1."
        )

    if (
        frequency_resolution_hz is not None
        and frequency_resolution_hz <= 0
    ):
        raise ValueError(
            "frequency_resolution_hz must be greater than zero."
        )

    if relative_tolerance_fraction < 0:
        raise ValueError(
            "relative_tolerance_fraction must be non-negative."
        )

    detected_frequencies = np.array(
        [
            peak.frequency_hz
            for peak in peaks.peaks
        ],
        dtype=float,
    )

    matches = []

    for harmonic_order in range(
        1,
        max_harmonic_order + 1,
    ):
        expected_frequency_hz = (
            harmonic_order
            * candidate_fundamental_hz
        )

        if detected_frequencies.size == 0:
            continue

        if frequency_resolution_hz is None:
            allowed_error_hz = tolerance_hz
        else:
            resolution_aware_tolerance = max(
                frequency_resolution_hz,
                (
                    relative_tolerance_fraction
                    * expected_frequency_hz
                ),
            )

            # tolerance_hz acts as an upper safety cap.
            allowed_error_hz = min(
                tolerance_hz,
                resolution_aware_tolerance,
            )

        nearest_index = int(
            np.argmin(
                np.abs(
                    detected_frequencies
                    - expected_frequency_hz
                )
            )
        )

        detected_frequency_hz = float(
            detected_frequencies[nearest_index]
        )

        frequency_error_hz = abs(
            detected_frequency_hz
            - expected_frequency_hz
        )

        if frequency_error_hz <= allowed_error_hz:
            matches.append(
                HarmonicMatch(
                    harmonic_order=harmonic_order,
                    expected_frequency_hz=float(
                        expected_frequency_hz
                    ),
                    detected_frequency_hz=detected_frequency_hz,
                    frequency_error_hz=float(
                        frequency_error_hz
                    ),
                    allowed_error_hz=float(
                        allowed_error_hz
                    ),
                )
            )

    matched_orders = [
        match.harmonic_order
        for match in matches
    ]

    if detected_frequencies.size > 0:
        highest_detected_frequency_hz = float(
            np.max(detected_frequencies)
        )

        considered_harmonic_orders = min(
            max_harmonic_order,
            int(
                np.floor(
                    highest_detected_frequency_hz
                    / candidate_fundamental_hz
                )
            ),
        )
    else:
        considered_harmonic_orders = 0

    if considered_harmonic_orders > 0:
        match_fraction = (
            len(matches)
            / considered_harmonic_orders
        )
    else:
        match_fraction = 0.0

    longest_consecutive_run = 0
    current_run = 0
    previous_order = None

    for order in matched_orders:
        if (
            previous_order is not None
            and order == previous_order + 1
        ):
            current_run += 1
        else:
            current_run = 1

        longest_consecutive_run = max(
            longest_consecutive_run,
            current_run,
        )

        previous_order = order

    return HarmonicFamilyResult(
        candidate_fundamental_hz=float(
            candidate_fundamental_hz
        ),
        matches=tuple(matches),
        tolerance_hz=float(tolerance_hz),
        match_fraction=float(match_fraction),
        longest_consecutive_run=longest_consecutive_run,
    )

@dataclass(frozen=True)
class AutomaticHarmonicSearchResult:
    best_family: HarmonicFamilyResult | None
    candidate_fundamentals_hz: tuple[float, ...]


def find_best_harmonic_family(
    peaks: PeakDetectionResult,
    tolerance_hz: float,
    max_harmonic_order: int = 10,
    minimum_matches: int = 3,
    frequency_resolution_hz: float | None = None,
    relative_tolerance_fraction: float = 0.01,
    minimum_match_fraction: float = 0.5,
    minimum_consecutive_matches: int = 3,
) -> AutomaticHarmonicSearchResult:
    """Search candidate fundamentals derived from peak spacings."""

    if tolerance_hz < 0:
        raise ValueError(
            "tolerance_hz must be non-negative."
        )

    if max_harmonic_order < 1:
        raise ValueError(
            "max_harmonic_order must be at least 1."
        )

    if minimum_matches < 2:
        raise ValueError(
            "minimum_matches must be at least 2."
        )

    if not 0.0 <= minimum_match_fraction <= 1.0:
        raise ValueError(
            "minimum_match_fraction must be between 0 and 1."
        )

    if minimum_consecutive_matches < 1:
        raise ValueError(
            "minimum_consecutive_matches must be at least 1."
        )

    detected_frequencies = np.array(
        sorted(
            peak.frequency_hz
            for peak in peaks.peaks
        ),
        dtype=float,
    )

    if detected_frequencies.size < 2:
        return AutomaticHarmonicSearchResult(
            best_family=None,
            candidate_fundamentals_hz=(),
        )

    candidate_fundamentals = set()

    for i in range(detected_frequencies.size):
        for j in range(i + 1, detected_frequencies.size):
            spacing_hz = (
                detected_frequencies[j]
                - detected_frequencies[i]
            )

            if spacing_hz > 0:
                candidate_fundamentals.add(
                    float(spacing_hz)
                )

    sorted_candidates = tuple(
        sorted(candidate_fundamentals)
    )

    best_family = None

    for candidate_hz in sorted_candidates:
        family = detect_harmonic_family(
            peaks,
            candidate_fundamental_hz=candidate_hz,
            tolerance_hz=tolerance_hz,
            max_harmonic_order=max_harmonic_order,
            frequency_resolution_hz=frequency_resolution_hz,
            relative_tolerance_fraction=(
                relative_tolerance_fraction
            ),
        )

        if len(family.matches) < minimum_matches:
            continue

        if (
            family.match_fraction
            < minimum_match_fraction
        ):
            continue

        if (
            family.longest_consecutive_run
            < minimum_consecutive_matches
        ):
            continue

        if best_family is None:
            best_family = family
            continue

        if len(family.matches) > len(best_family.matches):
            best_family = family
            continue

        if (
            len(family.matches)
            == len(best_family.matches)
            and family.longest_consecutive_run
            > best_family.longest_consecutive_run
        ):
            best_family = family
            continue

        if (
            len(family.matches)
            == len(best_family.matches)
            and family.longest_consecutive_run
            == best_family.longest_consecutive_run
            and family.candidate_fundamental_hz
            < best_family.candidate_fundamental_hz
        ):
            best_family = family

    return AutomaticHarmonicSearchResult(
        best_family=best_family,
        candidate_fundamentals_hz=sorted_candidates,
    )