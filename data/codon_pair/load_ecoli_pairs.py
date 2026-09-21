from pathlib import Path
import sys

import pandas as pd


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

DATASET = (
    PROJECT_ROOT
    / "data"
    / "codon_pair"
    / "ecoli_codon_pairs.xls"
)


# ---------------------------------------------------------
# Inspect workbook
# ---------------------------------------------------------

def inspect_workbook():

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET}"
        )

    excel = pd.ExcelFile(DATASET)

    print()
    print("=" * 70)
    print("E. COLI CODON-PAIR DATASET")
    print("=" * 70)

    print()
    print("File:")
    print(DATASET)

    print()
    print("Sheets:")
    print(excel.sheet_names)

    print()

    for sheet in excel.sheet_names:

        print("=" * 70)
        print(f"SHEET: {sheet}")
        print("=" * 70)

        df = pd.read_excel(
            DATASET,
            sheet_name=sheet,
            header=None,
        )

        print()
        print("Shape:")
        print(df.shape)

        print()
        print("Contents:")
        print(
            df.head(20).to_string(
                index=False,
                header=False
            )
        )

        print()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    if "--inspect" in sys.argv:
        inspect_workbook()

    else:
        print(
            "Use:\n"
            "python -m data.codon_pair.load_ecoli_pairs --inspect"
        )