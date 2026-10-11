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