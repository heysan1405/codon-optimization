from pathlib import Path
import sys
import json

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# FILES
# ============================================================

DATASET = (
    PROJECT_ROOT
    / "data"
    / "codon_pair"
    / "ecoli_codon_pairs.xls"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "data"
    / "codon_pair"
    / "ecoli_codon_pair_table.json"
)


# ============================================================
# CODON VALIDATION
# ============================================================

VALID_BASES = set("ACGT")


def is_valid_codon(codon: str) -> bool:
    return (
        isinstance(codon, str)
        and len(codon) == 3
        and set(codon).issubset(VALID_BASES)
    )


# ============================================================
# LOAD SHEET 2a
# ============================================================

def load_ecoli_codon_pairs():
    """
    Load experimentally observed E. coli codon-pair counts.

    Sheet 2a format:

        codon pairs | num codon | aa string | ...

    Example:

        CCTAGG | 1
        CCTAGA | 1
        CTAGGA | 10

    A six-nucleotide string is split into:

        CCT + AGG
        CCT + AGA
        CTA + GGA
    """

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET}"
        )

    df = pd.read_excel(
        DATASET,
        sheet_name="2a",
        header=None,
    )

    table = {}

    for _, row in df.iterrows():

        if len(row) < 2:
            continue

        raw_pair = row.iloc[0]
        raw_frequency = row.iloc[1]

        if pd.isna(raw_pair):
            continue

        pair_string = str(raw_pair).strip().upper()

        # Remove accidental whitespace.
        pair_string = pair_string.replace(" ", "")

        # Must contain exactly two codons.
        if len(pair_string) != 6:
            continue

        codon1 = pair_string[:3]
        codon2 = pair_string[3:]

        if not is_valid_codon(codon1):
            continue

        if not is_valid_codon(codon2):
            continue

        try:
            frequency = float(raw_frequency)
        except (TypeError, ValueError):
            continue

        if frequency < 0:
            continue

        table[(codon1, codon2)] = frequency

    if not table:
        raise ValueError(
            "No codon-pair observations were extracted "
            "from sheet 2a."
        )

    return table


# ============================================================
# VALIDATE
# ============================================================

def validate_table(table):

    if not isinstance(table, dict):
        raise TypeError(
            "Codon-pair table must be a dictionary."
        )

    for pair, frequency in table.items():

        if not isinstance(pair, tuple):
            raise TypeError(
                f"Invalid pair: {pair}"
            )

        if len(pair) != 2:
            raise ValueError(
                f"Invalid pair: {pair}"
            )

        codon1, codon2 = pair

        if not is_valid_codon(codon1):
            raise ValueError(
                f"Invalid codon: {codon1}"
            )

        if not is_valid_codon(codon2):
            raise ValueError(
                f"Invalid codon: {codon2}"
            )

        if frequency < 0:
            raise ValueError(
                f"Negative frequency: {pair}"
            )


# ============================================================
# SAVE JSON
# ============================================================

def save_table(table):

    output = {}

    for (codon1, codon2), frequency in table.items():

        key = f"{codon1}_{codon2}"

        output[key] = frequency

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    return OUTPUT_JSON


# ============================================================
# SUMMARY
# ============================================================

def print_summary(table):

    frequencies = list(table.values())

    print()
    print("=" * 70)
    print("E. COLI CODON-PAIR TABLE")
    print("=" * 70)

    print()
    print("Dataset:")
    print(DATASET)

    print()
    print("Source sheet:")
    print("2a")

    print()
    print("Codon pairs extracted:")
    print(len(table))

    print()
    print("Minimum observed count:")
    print(min(frequencies))

    print()
    print("Maximum observed count:")
    print(max(frequencies))

    print()
    print("First 20 pairs:")

    for (codon1, codon2), frequency in list(table.items())[:20]:

        print(
            f"{codon1} {codon2} -> {frequency}"
        )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    table = load_ecoli_codon_pairs()

    validate_table(table)

    print_summary(table)

    output = save_table(table)

    print("=" * 70)
    print("TABLE SAVED")
    print("=" * 70)
    print()
    print(output)
    print()


if __name__ == "__main__":
    main()