"""
K-mer constraint validation experiment.

Tests whether increasing the K-mer constraint changes the
optimized sequence while preserving translation.

Output:
    results/csv/kmer_validation.csv
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

KMER_SIZE = 6

# Deliberately include thresholds that should activate
# the constraint.
KMER_THRESHOLDS = [
    0.020,
    0.010,
    0.007,
    0.005,
    0.003,
    0.001,
]

KMER_WEIGHT = 20.0

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
    / "kmer_validation.csv"
)


# ============================================================
# FASTA LOADER
# ============================================================

def load_fasta(path: Path) -> str:

    if not path.exists():
        raise FileNotFoundError(
            f"FASTA file not found: {path}"
        )

    sequence = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):
            continue

        sequence.append(line)

    sequence = "".join(sequence).upper()

    if not sequence:
        raise ValueError(
            f"No DNA sequence found in {path}"
        )

    if not set(sequence).issubset(
        set("ACGT")
    ):
        raise ValueError(
            f"Invalid DNA characters in {path}"
        )

    if len(sequence) % 3 != 0:
        raise ValueError(
            f"Sequence length is not divisible "
            f"by 3: {path}"
        )

    return sequence


# ============================================================
# K-MER METRICS
# ============================================================

def calculate_kmer_metrics(
    sequence: str,
    k: int,
):
    """
    Calculate:

        maximum k-mer frequency
        mean k-mer frequency
        number of unique k-mers
    """

    sequence = sequence.upper()

    if len(sequence) < k:
        return 0.0, 0.0, 0

    counts = {}

    total = len(sequence) - k + 1

    for i in range(total):

        kmer = sequence[i:i + k]

        counts[kmer] = (
            counts.get(kmer, 0) + 1
        )

    frequencies = [
        count / total
        for count in counts.values()
    ]

    max_frequency = max(frequencies)

    mean_frequency = sum(
        frequencies
    ) / len(frequencies)

    unique_kmers = len(counts)

    return (
        max_frequency,
        mean_frequency,
        unique_kmers,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("K-MER CONSTRAINT VALIDATION")
    print("=" * 75)

    print(f"Organism          : {ORGANISM}")
    print(f"K-mer size        : {KMER_SIZE}")
    print(f"K-mer weight      : {KMER_WEIGHT}")
    print(f"Population        : {POPULATION_SIZE}")
    print(f"Generations       : {GENERATIONS}")
    print(f"Seed              : {SEED}")

    print(
        f"Thresholds        : "
        f"{KMER_THRESHOLDS}"
    )

    print("=" * 75)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    total_runs = (
        len(GENES)
        * len(KMER_THRESHOLDS)
    )

    run_number = 0

    # ========================================================
    # PROCESS GENES
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

        (
            original_max_kmer,
            original_mean_kmer,
            original_unique_kmers,
        ) = calculate_kmer_metrics(
            original,
            KMER_SIZE,
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
            f"Original CAI: "
            f"{original_cai:.6f}"
        )

        print(
            f"Original max "
            f"{KMER_SIZE}-mer frequency: "
            f"{original_max_kmer:.6f}"
        )

        print(
            f"Unique {KMER_SIZE}-mers: "
            f"{original_unique_kmers}"
        )

        print("-" * 75)

        # ====================================================
        # THRESHOLDS
        # ====================================================

        for threshold in KMER_THRESHOLDS:

            run_number += 1

            print(
                f"\n[{run_number}/{total_runs}] "
                f"{gene_name} | "
                f"threshold={threshold}"
            )

            start = time.perf_counter()

            result = optimize_ga(

                dna_sequence=original,

                organism=ORGANISM,

                # --------------------------------------------
                # Existing GC constraints
                # --------------------------------------------

                gc_min=0.40,
                gc_max=0.60,

                # --------------------------------------------
                # K-mer constraint
                # --------------------------------------------

                kmer_size=KMER_SIZE,

                max_kmer_frequency=threshold,

                kmer_weight=KMER_WEIGHT,

                # --------------------------------------------
                # GA
                # --------------------------------------------

                population_size=POPULATION_SIZE,

                generations=GENERATIONS,

                mutation_rate=MUTATION_RATE,

                crossover_rate=CROSSOVER_RATE,

                tournament_size=TOURNAMENT_SIZE,

                elitism=ELITISM,

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

            (
                optimized_max_kmer,
                optimized_mean_kmer,
                optimized_unique_kmers,
            ) = calculate_kmer_metrics(
                optimized,
                KMER_SIZE,
            )

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
                for a, b
                in zip(
                    original,
                    optimized,
                )
            )

            edit_rate = (
                edit_count
                / len(original)
            )

            # -------------------------------------------------
            # Threshold violation
            # -------------------------------------------------

            violation = max(
                0.0,
                optimized_max_kmer
                - threshold,
            )

            # -------------------------------------------------
            # Print
            # -------------------------------------------------

            print(
                f"  Max k-mer: "
                f"{original_max_kmer:.6f}"
                f" -> "
                f"{optimized_max_kmer:.6f}"
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
                f"  Violation: "
                f"{violation:.6f}"
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

                "kmer_size":
                    KMER_SIZE,

                "kmer_threshold":
                    threshold,

                "kmer_weight":
                    KMER_WEIGHT,

                "original_max_kmer_frequency":
                    original_max_kmer,

                "optimized_max_kmer_frequency":
                    optimized_max_kmer,

                "kmer_violation":
                    violation,

                "original_mean_kmer_frequency":
                    original_mean_kmer,

                "optimized_mean_kmer_frequency":
                    optimized_mean_kmer,

                "original_unique_kmers":
                    original_unique_kmers,

                "optimized_unique_kmers":
                    optimized_unique_kmers,

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
            "No experiment results were generated."
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

    print()
    print("=" * 75)
    print("K-MER VALIDATION COMPLETE")
    print("=" * 75)

    print(
        f"Runs completed : {len(rows)}"
    )

    print(
        f"Expected runs  : {total_runs}"
    )

    print(
        f"CSV saved to   : {OUTPUT}"
    )

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