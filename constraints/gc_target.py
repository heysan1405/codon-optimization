"""
GC-target optimization utilities.

Provides functions for measuring how far a DNA sequence
is from a desired GC-content target.
"""

from core.sequence import gc_content


# Small value used to handle floating-point rounding.
FLOAT_EPSILON = 1e-12


def validate_target_gc(target_gc):
    """
    Validate a GC target expressed as a fraction.

    Examples:
        0.50 = 50%
        0.60 = 60%
    """

    if not isinstance(target_gc, (int, float)):
        raise ValueError("target_gc must be numeric.")

    if not 0.0 <= target_gc <= 1.0:
        raise ValueError(
            "target_gc must be between 0.0 and 1.0."
        )

    return float(target_gc)


def gc_deviation(sequence, target_gc):
    """
    Calculate absolute deviation from the target GC fraction.

    Example:
        actual GC = 0.65
        target GC = 0.60
        deviation = 0.05
    """

    target_gc = validate_target_gc(target_gc)

    actual_gc = gc_content(sequence)

    return abs(actual_gc - target_gc)


def gc_target_score(sequence, target_gc):
    """
    Convert GC deviation into a similarity score.

    Returns:
        1.0 -> exact target
        0.0 -> maximum possible deviation

    Formula:
        score = 1 - |actual_GC - target_GC|
    """

    deviation = gc_deviation(
        sequence,
        target_gc,
    )

    return 1.0 - deviation


def within_gc_tolerance(
    sequence,
    target_gc,
    tolerance,
):
    """
    Check whether sequence GC is within the requested
    tolerance around the target.

    Example:
        target = 0.60
        tolerance = 0.05

        valid range = 0.55 to 0.65
    """

    target_gc = validate_target_gc(target_gc)

    if not isinstance(tolerance, (int, float)):
        raise ValueError(
            "tolerance must be numeric."
        )

    if tolerance < 0:
        raise ValueError(
            "tolerance cannot be negative."
        )

    actual_gc = gc_content(sequence)

    deviation = abs(
        actual_gc - target_gc
    )

    return deviation <= (
        float(tolerance) + FLOAT_EPSILON
    )