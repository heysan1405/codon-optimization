"""
Sequence similarity and edit-distance utilities.

These functions measure how much an optimized DNA sequence
differs from its original sequence.
"""


def validate_sequences(sequence1, sequence2):
    """Validate that two DNA sequences are comparable."""

    if not isinstance(sequence1, str) or not isinstance(sequence2, str):
        raise ValueError("Sequences must be strings.")

    sequence1 = sequence1.upper()
    sequence2 = sequence2.upper()

    if not sequence1 or not sequence2:
        raise ValueError("Sequences cannot be empty.")

    if any(base not in "ATGC" for base in sequence1):
        raise ValueError("Sequence 1 contains invalid DNA bases.")

    if any(base not in "ATGC" for base in sequence2):
        raise ValueError("Sequence 2 contains invalid DNA bases.")

    return sequence1, sequence2


def hamming_distance(sequence1, sequence2):
    """
    Calculate nucleotide-level Hamming distance.

    Both sequences must have the same length.
    """

    sequence1, sequence2 = validate_sequences(
        sequence1,
        sequence2,
    )

    if len(sequence1) != len(sequence2):
        raise ValueError(
            "Hamming distance requires sequences of equal length."
        )

    return sum(
        base1 != base2
        for base1, base2 in zip(sequence1, sequence2)
    )


def sequence_identity(sequence1, sequence2):
    """
    Calculate nucleotide sequence identity.

    Returns a value between 0 and 1.
    """

    sequence1, sequence2 = validate_sequences(
        sequence1,
        sequence2,
    )

    if len(sequence1) != len(sequence2):
        raise ValueError(
            "Sequence identity requires sequences of equal length."
        )

    matches = sum(
        base1 == base2
        for base1, base2 in zip(sequence1, sequence2)
    )

    return matches / len(sequence1)


def changed_positions(sequence1, sequence2):
    """
    Return zero-based nucleotide positions that changed.
    """

    sequence1, sequence2 = validate_sequences(
        sequence1,
        sequence2,
    )

    if len(sequence1) != len(sequence2):
        raise ValueError(
            "Changed positions require sequences of equal length."
        )

    return [
        position
        for position, (base1, base2)
        in enumerate(zip(sequence1, sequence2))
        if base1 != base2
    ]


def codon_changes(sequence1, sequence2):
    """
    Count codons that differ between two DNA sequences.

    Both sequences must contain complete codons and have
    the same length.
    """

    sequence1, sequence2 = validate_sequences(
        sequence1,
        sequence2,
    )

    if len(sequence1) != len(sequence2):
        raise ValueError(
            "Codon comparison requires equal-length sequences."
        )

    if len(sequence1) % 3 != 0:
        raise ValueError(
            "Sequences must have a length divisible by 3."
        )

    changes = 0

    for position in range(0, len(sequence1), 3):

        codon1 = sequence1[position:position + 3]
        codon2 = sequence2[position:position + 3]

        if codon1 != codon2:
            changes += 1

    return changes


def total_codons(sequence):
    """Return the number of complete codons in a DNA sequence."""

    if not isinstance(sequence, str):
        raise ValueError("Sequence must be a string.")

    sequence = sequence.upper()

    if not sequence:
        raise ValueError("Sequence cannot be empty.")

    if len(sequence) % 3 != 0:
        raise ValueError(
            "Sequence length must be divisible by 3."
        )

    return len(sequence) // 3


def edit_rate(sequence1, sequence2):
    """
    Calculate the fraction of codons changed.

    Example:
        2 changed codons / 10 total codons = 0.20
    """

    changes = codon_changes(
        sequence1,
        sequence2,
    )

    total = total_codons(sequence1)

    return changes / total