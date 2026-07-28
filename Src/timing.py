from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class TimingStatistics(Generic[T]):
    output: T
    samples_ms: tuple[float, ...]
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float
    coefficient_variation_percent: float


def measure(
    function: Callable[[], T],
    warmups: int,
    repetitions: int,
    synchronize: Callable[[], None] | None = None,
) -> TimingStatistics[T]:
    if repetitions < 1:
        raise ValueError("At least one measured repetition is required.")

    output: T | None = None
    for _ in range(warmups):
        output = function()
        if synchronize is not None:
            synchronize()

    samples: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        output = function()
        if synchronize is not None:
            synchronize()
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        samples.append(elapsed_ms)

    if output is None:
        output = function()
        if synchronize is not None:
            synchronize()

    mean_ms = statistics.fmean(samples)
    stddev_ms = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    coefficient = (stddev_ms / mean_ms * 100.0) if mean_ms > 0.0 else 0.0

    return TimingStatistics(
        output=output,
        samples_ms=tuple(samples),
        mean_ms=mean_ms,
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        stddev_ms=stddev_ms,
        coefficient_variation_percent=coefficient,
    )
