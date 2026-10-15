"""
Codon Adaptation Index (CAI) calculation.
"""

import math

from core.sequence import get_codons
from data.codon_tables import get_codon_table


def calculate_cai(sequence, organism):
    """
    Calculate the Codon Adaptation Index (CAI).

    CAI is calculated using the geometric mean of
    relative synonymous codon usage values.
    """

    codons = get_codons(sequence)

    table = get_codon_table(organism)

    relative_values = []

    for codon in codons:

        codon = codon.upper()

        amino_acid = None

        for aa, synonymous in table.items():

            if codon in synonymous:
                amino_acid = aa
                break

        if amino_acid is None:
            raise ValueError(
                f"Codon '{codon}' not found."
            )

        synonymous_codons = table[amino_acid]

        maximum_frequency = max(
            synonymous_codons.values()
        )

        codon_frequency = synonymous_codons[codon]

        relative_value = (
            codon_frequency / maximum_frequency
        )

        relative_values.append(relative_value)

    if not relative_values:
        return 0.0

    # Avoid log(0)
    epsilon = 1e-10

    log_sum = sum(
        math.log(max(value, epsilon))
        for value in relative_values
    )

    cai = math.exp(
        log_sum / len(relative_values)
    )

    return cai


def calculate_cai_batch(sequences, organism, device="auto"):
    """Calculate CAI for equal-length DNA sequences using PyTorch."""
    import torch

    if not sequences:
        return []

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    codon_rows = [get_codons(sequence) for sequence in sequences]
    if len({len(row) for row in codon_rows}) != 1:
        raise ValueError("All sequences must have equal codon lengths.")

    table = get_codon_table(organism)
    relative = {}

    for synonymous in table.values():
        maximum = max(synonymous.values())
        for codon, frequency in synonymous.items():
            relative[codon.upper()] = frequency / maximum

    codons = list(relative)
    codon_index = {codon: i for i, codon in enumerate(codons)}

    try:
        indices = [
            [codon_index[c.upper()] for c in row]
            for row in codon_rows
        ]
    except KeyError as exc:
        raise ValueError(f"Codon {exc.args[0]!r} not found.") from exc

    values = torch.tensor(
        [relative[c] for c in codons],
        dtype=torch.float64,
        device=device,
    )
    index_tensor = torch.tensor(indices, dtype=torch.long, device=device)

    result = torch.exp(
        torch.log(values[index_tensor].clamp_min(1e-10)).mean(dim=1)
    )
    return result.cpu().tolist()
