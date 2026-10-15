"""
Codon-pair bias validation experiment.

Purpose
-------
Test whether increasing codon-pair bias weight changes the optimized
sequence and improves the codon-pair score while preserving:

    1. Protein sequence
    2. Reasonable CAI
    3. GC constraints

Outputs
-------
results/csv/codon_pair_validation.csv
"""

from pathlib import Path
import csv
import sys
import time


# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from optimizers.genetic_algorithm import optimize_ga
from core.cai import calculate_cai
from core.codon_pair import codon_pair_score
from core.verification import translate_dna
from core.verification import verify_translation


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

HOST = "ecoli"

POPULATION_SIZE = 10
GENERATIONS = 3
MUTATION_RATE = 0.02
CROSSOVER_RATE = 0.80
TOURNAMENT_SIZE = 3
ELITISM = 2

# Test increasing codon-pair influence.
CODON_PAIR_WEIGHTS = [0.0]

# Reproducible experiment.
SEED = 42


# ---------------------------------------------------------------------
# Input genes
# ---------------------------------------------------------------------

GENES = {
    "GFP": PROJECT_ROOT / "data" / "genes" / "fasta" / "gene01_gfp.fasta",
    "Firefly_luciferase": PROJECT_ROOT / "data" / "genes" / "fasta" / "gene02_luc.fasta",
    "HBB": PROJECT_ROOT / "data" / "genes" / "fasta" / "gene03_hbb.fasta",
    "INS": PROJECT_ROOT / "data" / "genes" / "fasta" / "gene04_ins.fasta",
    "LYZ": PROJECT_ROOT / "data" / "genes" / "fasta" / "gene05_lyz.fasta",
    "ACTB": PROJECT_ROOT / "data" / "genes" / "fasta" / "gene06_actb.fasta",
    "TP53": PROJECT_ROOT / "data" / "genes" / "fasta" / "gene07_tp53.fasta",
    "LACZ": PROJECT_ROOT / "data" / "genes" / "fasta" / "gene08_lacz.fasta",
    "GROL": PROJECT_ROOT / "data" / "genes" / "fasta" / "gene09_grol.fasta",
    "SODA": PROJECT_ROOT / "data" / "genes" / "fasta" / "gene10_soda.fasta",
}


# ---------------------------------------------------------------------
# Codon-pair table
# ---------------------------------------------------------------------

PAIR_TABLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "codon_pair"
    / "ecoli_codon_pairs.xls"
)


def load_fasta(path: Path) -> str:
    """Load the first DNA sequence from a FASTA file."""

    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()

    sequence_parts = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):
            continue

        sequence_parts.append(line)

    sequence = "".join(sequence_parts).upper()

    valid = set("ACGT")

    if not sequence:
        raise ValueError(f"No sequence found in {path}")

    if not set(sequence).issubset(valid):
        bad = sorted(set(sequence) - valid)
        raise ValueError(
            f"Invalid DNA characters in {path}: {bad}"
        )

    if len(sequence) % 3 != 0:
        raise ValueError(
            f"Sequence length is not divisible by 3: "
            f"{path} ({len(sequence)} bp)"
        )

    return sequence


def load_pair_table():
    """
    Load the experimentally observed E. coli codon-pair table.

    This calls the project's existing loader instead of reconstructing
    pair frequencies from single-codon frequencies.
    """

    loader_path = (
        PROJECT_ROOT
        / "data"
        / "codon_pair"
        / "load_ecoli_pairs.py"
    )

    if not loader_path.exists():
        raise FileNotFoundError(
            f"Codon-pair loader not found: {loader_path}"
        )

    # Import the loader as a module.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "load_ecoli_pairs",
        loader_path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Try common loader function names.
    possible_names = [
        "load_ecoli_codon_pairs",
        "load_ecoli_pairs",
        "load_codon_pairs",
        "build_ecoli_pair_table",
    ]

    for name in possible_names:
        function = getattr(module, name, None)

        if callable(function):
            return function()

    raise RuntimeError(
        "Could not find a codon-pair loading function in "
        "data/codon_pair/load_ecoli_pairs.py.\n\n"
        "Expected one of:\n"
        "  load_ecoli_codon_pairs\n"
        "  load_ecoli_pairs\n"
        "  load_codon_pairs\n"
        "  build_ecoli_pair_table"
    )


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "csv"
    / "codon_pair_benchmark.csv"
)


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    print("=" * 75)
    print("CODON-PAIR BIAS VALIDATION")
    print("=" * 75)

    print(f"Host             : {HOST}")
    print(f"Population       : {POPULATION_SIZE}")
    print(f"Generations      : {GENERATIONS}")
    print(f"Seed             : {SEED}")
    print(f"Pair weights     : {CODON_PAIR_WEIGHTS}")
    print(f"Pair table       : {PAIR_TABLE_PATH}")
    print(f"Output           : {OUTPUT_PATH}")
    print("=" * 75)

    # -------------------------------------------------------------
    # Load codon-pair table
    # -------------------------------------------------------------

    print("\nLoading E. coli codon-pair table...")

    pair_table = load_pair_table()

    if not isinstance(pair_table, dict):
        raise TypeError(
            "Codon-pair loader did not return a dictionary."
        )

    print(
        f"Loaded {len(pair_table):,} codon-pair entries."
    )

    # -------------------------------------------------------------
    # Prepare output directory
    # -------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    total_runs = (
        len(GENES)
        * len(CODON_PAIR_WEIGHTS)
    )

    run_number = 0

    # -------------------------------------------------------------
    # Run experiments
    # -------------------------------------------------------------

    for gene_name, fasta_path in GENES.items():

        sequence = load_fasta(fasta_path)

        original_protein = translate_dna(sequence)

        original_cai = calculate_cai(
            sequence,
            organism=HOST,
        )

        original_pair_score = codon_pair_score(
            sequence,
            pair_table,
        )

        print("\n" + "-" * 75)
        print(f"Gene: {gene_name}")
        print(f"Length: {len(sequence)} bp")
        print(f"Original CAI: {original_cai:.4f}")
        print(
            f"Original codon-pair score: "
            f"{original_pair_score:.6f}"
        )
        print("-" * 75)

        for pair_weight in CODON_PAIR_WEIGHTS:

            run_number += 1

            print(
                f"\n[{run_number}/{total_runs}] "
                f"{gene_name} | "
                f"codon_pair_weight={pair_weight}"
            )

            start = time.perf_counter()

            result = optimize_ga(
                dna_sequence=sequence,
                organism=HOST,

                # Existing global GC constraint.
                gc_min=0.40,
                gc_max=0.60,

                # Codon-pair constraint.
                codon_pair_table=pair_table,
                codon_pair_weight=pair_weight,

                # GA parameters.
                population_size=POPULATION_SIZE,
                generations=GENERATIONS,
                mutation_rate=MUTATION_RATE,
                crossover_rate=CROSSOVER_RATE,
                tournament_size=TOURNAMENT_SIZE,
                elitism=ELITISM,

                seed=SEED,
            )

            runtime = time.perf_counter() - start

            # -----------------------------------------------------
            # Handle both return formats.
            # -----------------------------------------------------

            if isinstance(result, dict):
                optimized = result["sequence"]
            else:
                optimized = result

            optimized = optimized.upper()

            # -----------------------------------------------------
            # Metrics
            # -----------------------------------------------------

            optimized_cai = calculate_cai(
                optimized,
                organism=HOST,
            )

            optimized_pair_score = codon_pair_score(
                optimized,
                pair_table,
            )

            verification = verify_translation(
                sequence,
                optimized,
            )

            protein_preserved = bool(
                verification["valid"]
            )

            edit_count = sum(
                a != b
                for a, b in zip(sequence, optimized)
            )

            edit_rate = (
                edit_count / len(sequence)
                if sequence
                else 0.0
            )

            print(
                f"  CAI: "
                f"{original_cai:.4f} -> "
                f"{optimized_cai:.4f}"
            )

            print(
                f"  Pair score: "
                f"{original_pair_score:.6f} -> "
                f"{optimized_pair_score:.6f}"
            )

            print(
                f"  Edit rate: "
                f"{edit_rate:.2%}"
            )

            print(
                f"  Protein preserved: "
                f"{protein_preserved}"
            )

            print(
                f"  Runtime: "
                f"{runtime:.3f} s"
            )

            # -----------------------------------------------------
            # Save result
            # -----------------------------------------------------

            rows.append({
                "gene": gene_name,
                "length_bp": len(sequence),

                "codon_pair_weight": pair_weight,

                "original_cai": original_cai,
                "optimized_cai": optimized_cai,
                "delta_cai": (
                    optimized_cai - original_cai
                ),

                "original_codon_pair_score":
                    original_pair_score,

                "optimized_codon_pair_score":
                    optimized_pair_score,

                "delta_codon_pair_score": (
                    optimized_pair_score
                    - original_pair_score
                ),

                "edit_rate": edit_rate,
                "protein_preserved":
                    protein_preserved,

                "runtime_seconds": runtime,
                "seed": SEED,
            })

            # Save completed trials immediately.
            fieldnames = list(rows[0].keys())
            with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Checkpoint saved: {len(rows)} trial(s)")


    # -------------------------------------------------------------
    # Write CSV
    # -------------------------------------------------------------

    fieldnames = list(rows[0].keys())

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print("\n" + "=" * 75)
    print("EXPERIMENT COMPLETE")
    print("=" * 75)

    print(f"Runs completed : {len(rows)}")
    print(f"CSV saved to   : {OUTPUT_PATH}")

    preserved = sum(
        row["protein_preserved"]
        for row in rows
    )

    print(
        f"Protein preserved: "
        f"{preserved}/{len(rows)}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()
