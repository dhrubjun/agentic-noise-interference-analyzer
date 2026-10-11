from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventInterval:
    """Detected event interval in feature-track coordinates."""

    start_index: int
    end_index: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    peak_index: int
    peak_time_s: float
    peak_score: float

@dataclass(frozen=True)
class EventMeasurement:
    """Numerical evidence describing one detected event."""

    start_time_s: float
    end_time_s: float
    support_duration_s: float

    peak_score: float
    peak_rms: float
    mean_rms: float

    baseline_rms: float
    rms_ratio_to_baseline: float


def detect_events_with_hysteresis(
    times_seconds: np.ndarray,
    scores: np.ndarray,
    start_threshold: float,
    end_threshold: float,
) -> list[EventInterval]:
    """Detect event intervals from a deviation-score track."""

    times = np.asarray(times_seconds, dtype=float)
    values = np.asarray(scores, dtype=float)

    if times.ndim != 1:
        raise ValueError(
            "times_seconds must be one-dimensional."
        )

    if values.ndim != 1:
        raise ValueError(
            "scores must be one-dimensional."
        )

    if times.size == 0:
        raise ValueError(
            "times_seconds must not be empty."
        )

    if values.size == 0:
        raise ValueError(
            "scores must not be empty."
        )

    if times.size != values.size:
        raise ValueError(
            "times_seconds and scores must have the same length."
        )

    if not np.all(np.isfinite(times)):
        raise ValueError(
            "times_seconds must contain only finite values."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "scores must contain only finite values."
        )

    if start_threshold <= end_threshold:
        raise ValueError(
            "start_threshold must be greater than end_threshold."
        )

    events: list[EventInterval] = []

    in_event = False
    start_index = 0

    for index, score in enumerate(values):
        if not in_event:
            if score >= start_threshold:
                in_event = True
                start_index = index
        else:
            if score < end_threshold:
                end_index = index - 1

                event_scores = values[
                    start_index:end_index + 1
                ]

                relative_peak_index = int(
                    np.argmax(event_scores)
                )

                peak_index = (
                    start_index + relative_peak_index
                )

                events.append(
                    EventInterval(
                        start_index=start_index,
                        end_index=end_index,
                        start_time_s=float(
                            times[start_index]
                        ),
                        end_time_s=float(
                            times[end_index]
                        ),
                        duration_s=float(
                            times[end_index]
                            - times[start_index]
                        ),
                        peak_index=peak_index,
                        peak_time_s=float(
                            times[peak_index]
                        ),
                        peak_score=float(
                            values[peak_index]
                        ),
                    )
                )

                in_event = False

    if in_event:
        end_index = values.size - 1

        event_scores = values[
            start_index:end_index + 1
        ]

        relative_peak_index = int(
            np.argmax(event_scores)
        )

        peak_index = (
            start_index + relative_peak_index
        )

        events.append(
            EventInterval(
                start_index=start_index,
                end_index=end_index,
                start_time_s=float(
                    times[start_index]
                ),
                end_time_s=float(
                    times[end_index]
                ),
                duration_s=float(
                    times[end_index]
                    - times[start_index]
                ),
                peak_index=peak_index,
                peak_time_s=float(
                    times[peak_index]
                ),
                peak_score=float(
                    values[peak_index]
                ),
            )
        )

    return events

def postprocess_events(
    events: list[EventInterval],
    frame_length_seconds: float,
    minimum_event_duration_s: float = 0.0,
    merge_gap_s: float = 0.0,
) -> list[EventInterval]:
    """Merge nearby events and reject events shorter than a configured duration."""

    if frame_length_seconds <= 0:
        raise ValueError(
            "frame_length_seconds must be greater than zero."
        )

    if minimum_event_duration_s < 0:
        raise ValueError(
            "minimum_event_duration_s must be non-negative."
        )

    if merge_gap_s < 0:
        raise ValueError(
            "merge_gap_s must be non-negative."
        )

    if not events:
        return []

    merged: list[EventInterval] = []

    current = events[0]

    for next_event in events[1:]:
        current_support_end = (
            current.end_time_s
            + frame_length_seconds / 2.0
        )

        next_support_start = (
            next_event.start_time_s
            - frame_length_seconds / 2.0
        )

        gap_s = (
            next_support_start
            - current_support_end
        )

        if gap_s <= merge_gap_s:
            if next_event.peak_score > current.peak_score:
                peak_index = next_event.peak_index
                peak_time_s = next_event.peak_time_s
                peak_score = next_event.peak_score
            else:
                peak_index = current.peak_index
                peak_time_s = current.peak_time_s
                peak_score = current.peak_score

            current = EventInterval(
                start_index=current.start_index,
                end_index=next_event.end_index,
                start_time_s=current.start_time_s,
                end_time_s=next_event.end_time_s,
                duration_s=(
                    next_event.end_time_s
                    - current.start_time_s
                ),
                peak_index=peak_index,
                peak_time_s=peak_time_s,
                peak_score=peak_score,
            )

        else:
            merged.append(current)
            current = next_event

    merged.append(current)

    filtered: list[EventInterval] = []

    for event in merged:
        support_duration_s = (
            event.end_time_s
            - event.start_time_s
            + frame_length_seconds
        )

        if support_duration_s >= minimum_event_duration_s:
            filtered.append(event)

    return filtered

def measure_rms_event(
    event: EventInterval,
    rms_values: np.ndarray,
    baseline_rms: float,
    frame_length_seconds: float,
) -> EventMeasurement:
    """Measure RMS-based evidence for one detected event."""

    values = np.asarray(
        rms_values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "rms_values must be one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            "rms_values must not be empty."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "rms_values must contain only finite values."
        )

    if not np.isfinite(baseline_rms):
        raise ValueError(
            "baseline_rms must be finite."
        )

    if baseline_rms <= 0:
        raise ValueError(
            "baseline_rms must be greater than zero."
        )

    if frame_length_seconds <= 0:
        raise ValueError(
            "frame_length_seconds must be greater than zero."
        )

    if event.start_index < 0:
        raise ValueError(
            "event start_index must be non-negative."
        )

    if event.end_index >= values.size:
        raise ValueError(
            "event end_index exceeds rms_values length."
        )

    if event.end_index < event.start_index:
        raise ValueError(
            "event end_index must not be smaller than start_index."
        )

    event_rms = values[
        event.start_index:event.end_index + 1
    ]

    peak_rms = float(
        np.max(event_rms)
    )

    mean_rms = float(
        np.mean(event_rms)
    )

    support_duration_s = float(
        event.end_time_s
        - event.start_time_s
        + frame_length_seconds
    )

    rms_ratio_to_baseline = float(
        peak_rms / baseline_rms
    )

    return EventMeasurement(
        start_time_s=event.start_time_s,
        end_time_s=event.end_time_s,
        support_duration_s=support_duration_s,
        peak_score=event.peak_score,
        peak_rms=peak_rms,
        mean_rms=mean_rms,
        baseline_rms=float(baseline_rms),
        rms_ratio_to_baseline=rms_ratio_to_baseline,
    )