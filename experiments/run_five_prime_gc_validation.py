"""
5'-GC constraint validation experiment.

Tests whether the optimizer can control GC content in the
5' region of coding sequences while preserving translation.

Output:
    results/csv/five_prime_gc_validation.csv
"""

from pathlib import Path
import csv
import sys
import time


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from optimizers.genetic_algorithm import optimize_ga
from core.cai import calculate_cai
from core.verification import translate_dna
from core.verification import verify_translation


# ============================================================
# CONFIGURATION
# ============================================================

ORGANISM = "ecoli"

POPULATION_SIZE = 50
GENERATIONS = 50

MUTATION_RATE = 0.02
CROSSOVER_RATE = 0.80

TOURNAMENT_SIZE = 3
ELITISM = 2

# First 60 nucleotides.
FIVE_PRIME_WINDOW = 60

# Test progressively stronger constraints.
GC_RANGES = [
    (0.35, 0.45),
    (0.40, 0.50),
    (0.45, 0.55),
    (0.50, 0.60),
    (0.55, 0.65),
]

FIVE_PRIME_GC_WEIGHT = 20.0

SEED = 42


# ============================================================
# GENES
# ============================================================

GENES = {
    "GFP":
        PROJECT_ROOT / "data" / "genes" / "fasta" /
        "gene01_gfp.fasta",

    "LUC":
        PROJECT_ROOT / "data" / "genes" / "fasta" /
        "gene02_luc.fasta",

    "HBB":
        PROJECT_ROOT / "data" / "genes" / "fasta" /
        "gene03_hbb.fasta",

    "INS":
        PROJECT_ROOT / "data" / "genes" / "fasta" /
        "gene04_ins.fasta",

    "LYZ":
        PROJECT_ROOT / "data" / "genes" / "fasta" /
        "gene05_lyz.fasta",

    "ACTB":
        PROJECT_ROOT / "data" / "genes" / "fasta" /
        "gene06_actb.fasta",

    "TP53":
        PROJECT_ROOT / "data" / "genes" / "fasta" /
        "gene07_tp53.fasta",

    "LACZ":
        PROJECT_ROOT / "data" / "genes" / "fasta" /
        "gene08_lacz.fasta",

    "GROL":
        PROJECT_ROOT / "data" / "genes" / "fasta" /
        "gene09_grol.fasta",

    "SODA":
        PROJECT_ROOT / "data" / "genes" / "fasta" /
        "gene10_soda.fasta",
}


# ============================================================
# OUTPUT
# ============================================================

OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "csv"
    / "five_prime_gc_validation.csv"
)


# ============================================================
# FASTA
# ============================================================

def load_fasta(path: Path) -> str:

    if not path.exists():
        raise FileNotFoundError(
            f"FASTA file not found: {path}"
        )

    parts = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):
            continue

        parts.append(line)

    sequence = "".join(parts).upper()

    if not sequence:
        raise ValueError(
            f"No DNA sequence found: {path}"
        )

    if not set(sequence).issubset(
        set("ACGT")
    ):
        raise ValueError(
            f"Invalid DNA characters: {path}"
        )

    if len(sequence) % 3 != 0:
        raise ValueError(
            f"Sequence length is not divisible by 3: "
            f"{path}"
        )

    return sequence


# ============================================================
# GC METRIC
# ============================================================

def gc_content(sequence: str) -> float:

    if not sequence:
        return 0.0

    sequence = sequence.upper()

    return (
        sequence.count("G")
        + sequence.count("C")
    ) / len(sequence)


def five_prime_gc(sequence: str) -> float:

    window = sequence[:FIVE_PRIME_WINDOW]

    return gc_content(window)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("5'-GC CONSTRAINT VALIDATION")
    print("=" * 75)

    print(f"Organism       : {ORGANISM}")
    print(
        f"5'-GC window   : "
        f"{FIVE_PRIME_WINDOW} bp"
    )
    print(
        f"GC ranges      : "
        f"{GC_RANGES}"
    )
    print(
        f"GC weight      : "
        f"{FIVE_PRIME_GC_WEIGHT}"
    )
    print(
        f"Population     : "
        f"{POPULATION_SIZE}"
    )
    print(
        f"Generations    : "
        f"{GENERATIONS}"
    )
    print(
        f"Seed           : "
        f"{SEED}"
    )

    print("=" * 75)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    total_runs = (
        len(GENES)
        * len(GC_RANGES)
    )

    run_number = 0

    # ========================================================
    # GENES
    # ========================================================

    for gene_name, fasta_path in GENES.items():

        original = load_fasta(
            fasta_path
        )

        original_protein = translate_dna(
            original
        )

        original_cai = calculate_cai(
            original,
            organism=ORGANISM,
        )

        original_gc = five_prime_gc(
            original
        )

        print()
        print("-" * 75)
        print(
            f"GENE: {gene_name}"
        )
        print(
            f"Length: {len(original)} bp"
        )
        print(
            f"Original 5'-GC: "
            f"{original_gc:.4f}"
        )
        print(
            f"Original CAI: "
            f"{original_cai:.6f}"
        )
        print("-" * 75)

        # ====================================================
        # GC TARGET RANGES
        # ====================================================

        for gc_min, gc_max in GC_RANGES:

            run_number += 1

            print(
                f"\n[{run_number}/{total_runs}] "
                f"{gene_name} | "
                f"5'-GC={gc_min:.2f}-"
                f"{gc_max:.2f}"
            )

            start = time.perf_counter()

            result = optimize_ga(

                dna_sequence=original,

                organism=ORGANISM,

                # Existing global GC.
                gc_min=0.40,
                gc_max=0.60,

                # 5'-GC constraint.
                five_prime_window=
                    FIVE_PRIME_WINDOW,

                five_prime_gc_min=gc_min,

                five_prime_gc_max=gc_max,

                five_prime_gc_weight=
                    FIVE_PRIME_GC_WEIGHT,

                # GA.
                population_size=
                    POPULATION_SIZE,

                generations=
                    GENERATIONS,

                mutation_rate=
                    MUTATION_RATE,

                crossover_rate=
                    CROSSOVER_RATE,

                tournament_size=
                    TOURNAMENT_SIZE,

                elitism=
                    ELITISM,

                seed=SEED,
            )

            runtime = (
                time.perf_counter()
                - start
            )

            # =================================================
            # RESULT
            # =================================================

            if isinstance(result, dict):
                optimized = result[
                    "sequence"
                ]
            else:
                optimized = result

            optimized = optimized.upper()

            # =================================================
            # METRICS
            # =================================================

            optimized_cai = calculate_cai(
                optimized,
                organism=ORGANISM,
            )

            optimized_gc = five_prime_gc(
                optimized
            )

            # -------------------------------------------------
            # Constraint violation
            # -------------------------------------------------

            if optimized_gc < gc_min:

                violation = (
                    gc_min
                    - optimized_gc
                )

            elif optimized_gc > gc_max:

                violation = (
                    optimized_gc
                    - gc_max
                )

            else:

                violation = 0.0

            # -------------------------------------------------
            # Protein preservation
            # -------------------------------------------------

            verification = verify_translation(
                original,
                optimized,
            )

            protein_preserved = bool(
                verification["valid"]
            )

            # -------------------------------------------------
            # Edit rate
            # -------------------------------------------------

            edit_count = sum(
                a != b
                for a, b in zip(
                    original,
                    optimized,
                )
            )

            edit_rate = (
                edit_count
                / len(original)
            )

            # -------------------------------------------------
            # Within target?
            # -------------------------------------------------

            within_target = (
                gc_min
                <= optimized_gc
                <= gc_max
            )

            # -------------------------------------------------
            # Print
            # -------------------------------------------------

            print(
                f"  5'-GC: "
                f"{original_gc:.4f}"
                f" -> "
                f"{optimized_gc:.4f}"
            )

            print(
                f"  Target: "
                f"{gc_min:.2f}-"
                f"{gc_max:.2f}"
            )

            print(
                f"  Violation: "
                f"{violation:.6f}"
            )

            print(
                f"  Within target: "
                f"{within_target}"
            )

            print(
                f"  CAI: "
                f"{original_cai:.6f}"
                f" -> "
                f"{optimized_cai:.6f}"
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
                f"{runtime:.3f} sec"
            )

            # =================================================
            # SAVE
            # =================================================

            rows.append({

                "gene":
                    gene_name,

                "length_bp":
                    len(original),

                "five_prime_window":
                    FIVE_PRIME_WINDOW,

                "five_prime_gc_min":
                    gc_min,

                "five_prime_gc_max":
                    gc_max,

                "five_prime_gc_weight":
                    FIVE_PRIME_GC_WEIGHT,

                "original_five_prime_gc":
                    original_gc,

                "optimized_five_prime_gc":
                    optimized_gc,

                "five_prime_gc_violation":
                    violation,

                "within_target":
                    within_target,

                "original_cai":
                    original_cai,

                "optimized_cai":
                    optimized_cai,

                "delta_cai":
                    optimized_cai
                    - original_cai,

                "edit_rate":
                    edit_rate,

                "protein_preserved":
                    protein_preserved,

                "runtime_seconds":
                    runtime,

                "seed":
                    SEED,
            })

    # ========================================================
    # WRITE CSV
    # ========================================================

    if not rows:
        raise RuntimeError(
            "No experiment results generated."
        )

    fieldnames = list(
        rows[0].keys()
    )

    with OUTPUT.open(
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

    # ========================================================
    # SUMMARY
    # ========================================================

    successful = sum(
        row["within_target"]
        for row in rows
    )

    preserved = sum(
        row["protein_preserved"]
        for row in rows
    )

    print()
    print("=" * 75)
    print("5'-GC VALIDATION COMPLETE")
    print("=" * 75)

    print(
        f"Runs completed : {len(rows)}"
    )

    print(
        f"Within target  : "
        f"{successful}/{len(rows)}"
    )

    print(
        f"Protein preserved: "
        f"{preserved}/{len(rows)}"
    )

    print(
        f"CSV saved to   : {OUTPUT}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()