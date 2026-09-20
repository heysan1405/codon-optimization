"""
Codon-pair bias utilities.

This module provides generic codon-pair scoring functions.
Pair-frequency data must be supplied separately because
single-codon usage frequencies cannot be used as a substitute
for experimentally observed codon-pair frequencies.
"""

from core.sequence import get_codons


def validate_pair_table(pair_table):
    """
    Validate a codon-pair frequency table.

    Expected format:

    {
        ("GCT", "GCC"): 0.12,
        ("GCC", "GCT"): 0.08,
        ...
    }

    Parameters
    ----------
    pair_table : dict
        Mapping of (codon1, codon2) -> frequency/score.

    Returns
    -------
    bool
        True if the table is valid.
    """

    if not isinstance(pair_table, dict):
        raise ValueError("pair_table must be a dictionary.")

    for pair, value in pair_table.items():

        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
        ):
            raise ValueError(
                "Each pair-table key must be "
                "a tuple containing two codons."
            )

        codon1, codon2 = pair

        if (
            not isinstance(codon1, str)
            or not isinstance(codon2, str)
        ):
            raise ValueError(
                "Codons in pair-table keys must be strings."
            )

        codon1 = codon1.upper()
        codon2 = codon2.upper()

        if len(codon1) != 3 or len(codon2) != 3:
            raise ValueError(
                "Each codon must contain exactly 3 bases."
            )

        if any(base not in "ATGC" for base in codon1 + codon2):
            raise ValueError(
                "Codons may contain only A, T, G and C."
            )

        if not isinstance(value, (int, float)):
            raise ValueError(
                "Pair frequencies/scores must be numeric."
            )

        if value < 0:
            raise ValueError(
                "Pair frequencies/scores cannot be negative."
            )

    return True


def get_codon_pairs(sequence):
    """
    Return consecutive codon pairs from a DNA sequence.

    Example
    -------
    ATGGCCGCG

    codons:
        ATG, GCC, GCG

    pairs:
        (ATG, GCC)
        (GCC, GCG)
    """

    codons = get_codons(sequence)

    return list(
        zip(codons[:-1], codons[1:])
    )


def codon_pair_frequency(
    codon1,
    codon2,
    pair_table,
):
    """
    Return the frequency/score of a codon pair.

    If the pair is not present in the supplied table,
    0.0 is returned.
    """

    validate_pair_table(pair_table)

    pair = (
        codon1.upper(),
        codon2.upper(),
    )

    return float(
        pair_table.get(pair, 0.0)
    )


def codon_pair_score(
    sequence,
    pair_table,
):
    """
    Calculate the mean codon-pair score of a DNA sequence.

    The score is the arithmetic mean of the supplied
    pair frequencies/scores across all consecutive
    codon pairs.

    A sequence containing fewer than two codons has
    no codon pairs and therefore returns 0.0.
    """

    pairs = get_codon_pairs(sequence)

    if not pairs:
        return 0.0

    scores = [
        codon_pair_frequency(
            codon1,
            codon2,
            pair_table,
        )
        for codon1, codon2 in pairs
    ]

    return sum(scores) / len(scores)


def codon_pair_scores(
    sequence,
    pair_table,
):
    """
    Return the individual score for every codon pair.

    Returns
    -------
    list of dict
        Each item contains:

        {
            "position": int,
            "codon1": str,
            "codon2": str,
            "score": float
        }
    """

    pairs = get_codon_pairs(sequence)

    results = []

    for position, (codon1, codon2) in enumerate(pairs):

        score = codon_pair_frequency(
            codon1,
            codon2,
            pair_table,
        )

        results.append(
            {
                "position": position,
                "codon1": codon1,
                "codon2": codon2,
                "score": score,
            }
        )

    return results