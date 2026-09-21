"""
Codon-pair frequency tables.

The table format is:

{
    "ecoli": {
        ("AAA", "AAA"): frequency,
        ("AAA", "AAG"): frequency,
        ...
    }
}

IMPORTANT:
These values must come from an experimentally observed codon-pair
usage dataset. Do not derive codon-pair frequencies from single-codon
frequencies.
"""

from typing import Dict, Tuple


CodonPairTable = Dict[Tuple[str, str], float]


def validate_codon_pair_table(table: CodonPairTable) -> None:
    """Validate a codon-pair frequency table."""

    if not isinstance(table, dict):
        raise TypeError("Codon-pair table must be a dictionary.")

    for pair, frequency in table.items():

        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(
                f"Invalid codon pair: {pair}. "
                "Expected ('AAA', 'AAA')."
            )

        codon1, codon2 = pair

        if not isinstance(codon1, str) or not isinstance(codon2, str):
            raise TypeError(
                f"Codons must be strings: {pair}"
            )

        if len(codon1) != 3 or len(codon2) != 3:
            raise ValueError(
                f"Invalid codon length: {pair}"
            )

        codon1 = codon1.upper()
        codon2 = codon2.upper()

        valid_bases = set("ACGT")

        if not set(codon1).issubset(valid_bases):
            raise ValueError(f"Invalid codon: {codon1}")

        if not set(codon2).issubset(valid_bases):
            raise ValueError(f"Invalid codon: {codon2}")

        if not isinstance(frequency, (int, float)):
            raise TypeError(
                f"Frequency must be numeric: {pair}"
            )

        if frequency < 0:
            raise ValueError(
                f"Frequency cannot be negative: {pair}"
            )


def get_codon_pair(
    codon1: str,
    codon2: str
) -> Tuple[str, str]:
    """Return a normalized codon pair."""

    codon1 = codon1.upper()
    codon2 = codon2.upper()

    return codon1, codon2


def get_pair_frequency(
    table: CodonPairTable,
    codon1: str,
    codon2: str
) -> float:
    """Return observed frequency of a codon pair."""

    pair = get_codon_pair(codon1, codon2)

    return float(table.get(pair, 0.0))


def codon_pair_score(
    table: CodonPairTable,
    codon1: str,
    codon2: str
) -> float:
    """
    Convert codon-pair frequency into a normalized score.

    Higher frequency -> higher score.

    The table is expected to contain normalized observed
    codon-pair frequencies.
    """

    frequency = get_pair_frequency(
        table,
        codon1,
        codon2
    )

    if frequency <= 0:
        return 0.0

    return frequency


def sequence_codon_pair_score(
    sequence: str,
    table: CodonPairTable
) -> float:
    """
    Calculate mean codon-pair score for a DNA sequence.
    """

    sequence = sequence.upper().replace(" ", "").replace("\n", "")

    if len(sequence) % 3 != 0:
        raise ValueError(
            "DNA sequence length must be divisible by 3."
        )

    codons = [
        sequence[i:i + 3]
        for i in range(0, len(sequence), 3)
    ]

    if len(codons) < 2:
        return 0.0

    scores = []

    for i in range(len(codons) - 1):

        score = codon_pair_score(
            table,
            codons[i],
            codons[i + 1]
        )

        scores.append(score)

    return sum(scores) / len(scores)