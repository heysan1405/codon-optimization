"""
Constraint Sensitivity Experiment

Tests the sensitivity of the GA to:
1. K-mer frequency threshold and penalty weight
2. 5' local GC penalty weight
3. Combined k-mer + 5' GC constraints

Dataset:
    10 biological benchmark genes in data/genes/fasta/

Output:
    results/csv/constraint_sensitivity.csv
"""

from pathlib import Path
import csv
import time
import sys
from collections import Counter


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from core.sequence import gc_content
from core.cai import calculate_cai
from core.similarity import (
    changed_positions,
    total_codons,
    edit_rate,
)
from core.verification import (
    translate_dna,
    verify_translation,
)
from optimizers.genetic_algorithm import optimize_ga


# ============================================================
# PATHS
# ============================================================

FASTA_DIR = PROJECT_ROOT / "data" / "genes" / "fasta"
OUTPUT_DIR = PROJECT_ROOT / "results" / "csv"

OUTPUT_FILE = (
    OUTPUT_DIR / "constraint_sensitivity.csv"
)


# ============================================================
# BIOLOGICAL BENCHMARK DATASET
# ============================================================

GENES = {
    "GFP": "gene01_gfp.fasta",
    "Firefly_luciferase": "gene02_luc.fasta",
    "HBB": "gene03_hbb.fasta",
    "INS": "gene04_ins.fasta",
    "LYZ": "gene05_lyz.fasta",
    "ACTB": "gene06_actb.fasta",
    "TP53": "gene07_tp53.fasta",
    "lacZ": "gene08_lacz.fasta",
    "groL": "gene09_grol.fasta",
    "sodA": "gene10_soda.fasta",
}


# ============================================================
# GA SETTINGS
# ============================================================

POPULATION_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.02
CROSSOVER_RATE = 0.80
TOURNAMENT_SIZE = 3
ELITISM = 2
SEED = 42

CAI_WEIGHT = 1.0
GC_WEIGHT = 10.0
TARGET_GC_WEIGHT = 10.0
LOCAL_GC_WEIGHT = 5.0
REPEAT_WEIGHT = 5.0
HOMOPOLYMER_WEIGHT = 5.0
RESTRICTION_WEIGHT = 5.0
EDIT_WEIGHT = 0.0


# ============================================================
# K-MER SETTINGS
# ============================================================

KMER_SIZE = 6

KMER_THRESHOLDS = [
    0.02,
    0.01,
    0.005,
]

KMER_WEIGHTS = [
    5.0,
    10.0,
    20.0,
]


# ============================================================
# 5' GC SETTINGS
# ============================================================

FIVE_PRIME_WINDOW = 60

FIVE_PRIME_GC_MIN = 0.40
FIVE_PRIME_GC_MAX = 0.60

FIVE_PRIME_WEIGHTS = [
    5.0,
    10.0,
    20.0,
]


# ============================================================
# COMBINED EXPERIMENTS
# ============================================================

COMBINED_CONFIGS = [
    {
        "name": "Combined_02_5_5",
        "kmer_threshold": 0.02,
        "kmer_weight": 5.0,
        "five_prime_weight": 5.0,
    },
    {
        "name": "Combined_01_10_10",
        "kmer_threshold": 0.01,
        "kmer_weight": 10.0,
        "five_prime_weight": 10.0,
    },
    {
        "name": "Combined_005_20_20",
        "kmer_threshold": 0.005,
        "kmer_weight": 20.0,
        "five_prime_weight": 20.0,
    },
    {
        "name": "Combined_01_20_20",
        "kmer_threshold": 0.01,
        "kmer_weight": 20.0,
        "five_prime_weight": 20.0,
    },
    {
        "name": "Combined_005_20_10",
        "kmer_threshold": 0.005,
        "kmer_weight": 20.0,
        "five_prime_weight": 10.0,
    },
]


# ============================================================
# FASTA READER
# ============================================================

def read_fasta(path):
    """Read a DNA FASTA sequence safely."""

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    sequence_parts = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):
            continue

        cleaned = "".join(
            base
            for base in line.upper()
            if base in "ACGT"
        )

        if cleaned:
            sequence_parts.append(cleaned)

    sequence = "".join(sequence_parts)

    if not sequence:
        raise ValueError(
            f"No DNA sequence found in FASTA file: {path}"
        )

    invalid = set(sequence) - set("ACGT")

    if invalid:
        raise ValueError(
            f"Invalid DNA bases found: {sorted(invalid)}"
        )

    return sequence


# ============================================================
# K-MER STATISTICS
# ============================================================

def get_max_kmer_stats(sequence, k=6):
    """
    Return the most frequent k-mer and its frequency.
    """

    if len(sequence) < k:
        return "", 0.0

    kmers = [
        sequence[i:i + k]
        for i in range(len(sequence) - k + 1)
    ]

    counts = Counter(kmers)

    most_common_kmer, count = (
        counts.most_common(1)[0]
    )

    frequency = count / len(kmers)

    return most_common_kmer, frequency


# ============================================================
# 5' GC
# ============================================================

def get_five_prime_gc(sequence, window=60):
    """
    Calculate GC fraction in the first window bases.
    """

    if not sequence:
        return 0.0

    region = sequence[:window]

    return gc_content(region)


# ============================================================
# CONFIGURATION LIST
# ============================================================

def build_experiments():
    """
    Build all 18 configurations.

    1 baseline
    9 k-mer experiments
    3 5' GC experiments
    5 combined experiments
    """

    experiments = []

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    experiments.append(
        {
            "name": "Baseline_GA",
            "type": "baseline",
            "kmer_threshold": None,
            "kmer_weight": 0.0,
            "five_prime_weight": 0.0,
        }
    )

    # --------------------------------------------------------
    # K-mer sensitivity
    # --------------------------------------------------------

    for threshold in KMER_THRESHOLDS:

        for weight in KMER_WEIGHTS:

            experiments.append(
                {
                    "name": (
                        f"Kmer_t{threshold}_w{weight}"
                    ),
                    "type": "kmer",
                    "kmer_threshold": threshold,
                    "kmer_weight": weight,
                    "five_prime_weight": 0.0,
                }
            )

    # --------------------------------------------------------
    # 5' GC sensitivity
    # --------------------------------------------------------

    for weight in FIVE_PRIME_WEIGHTS:

        experiments.append(
            {
                "name": f"FivePrimeGC_w{weight}",
                "type": "five_prime_gc",
                "kmer_threshold": None,
                "kmer_weight": 0.0,
                "five_prime_weight": weight,
            }
        )

    # --------------------------------------------------------
    # Combined sensitivity
    # --------------------------------------------------------

    for config in COMBINED_CONFIGS:

        experiments.append(
            {
                "name": config["name"],
                "type": "combined",
                "kmer_threshold": (
                    config["kmer_threshold"]
                ),
                "kmer_weight": (
                    config["kmer_weight"]
                ),
                "five_prime_weight": (
                    config["five_prime_weight"]
                ),
            }
        )

    return experiments


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_one(
    gene_name,
    sequence,
    config,
):

    # --------------------------------------------------------
    # Original sequence statistics
    # --------------------------------------------------------

    original_cai = calculate_cai(
        sequence,
        organism="ecoli",
    )

    original_gc = gc_content(
        sequence
    )

    # Keep this for reference / checking.
    original_protein = translate_dna(
        sequence
    )

    start = time.perf_counter()

    # --------------------------------------------------------
    # Run GA
    # --------------------------------------------------------

    optimized = optimize_ga(
        dna_sequence=sequence,
        organism="ecoli",

        # Existing constraints
        gc_min=None,
        gc_max=None,
        target_gc=None,

        local_gc_window=None,
        local_gc_min=None,
        local_gc_max=None,

        repeat_length=None,
        max_homopolymer=None,
        forbidden_sites=None,

        # GA
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        tournament_size=TOURNAMENT_SIZE,
        elitism=ELITISM,

        # Existing objective weights
        cai_weight=CAI_WEIGHT,
        gc_weight=GC_WEIGHT,
        target_gc_weight=TARGET_GC_WEIGHT,
        local_gc_weight=LOCAL_GC_WEIGHT,
        repeat_weight=REPEAT_WEIGHT,
        homopolymer_weight=HOMOPOLYMER_WEIGHT,
        restriction_weight=RESTRICTION_WEIGHT,
        edit_weight=EDIT_WEIGHT,

        # ----------------------------------------------------
        # Codon-pair bias
        # ----------------------------------------------------

        codon_pair_table=None,
        codon_pair_weight=0.0,

        # ----------------------------------------------------
        # K-mer constraint
        # ----------------------------------------------------

        kmer_size=(
            KMER_SIZE
            if config["kmer_weight"] > 0
            else None
        ),

        max_kmer_frequency=(
            config["kmer_threshold"]
            if config["kmer_weight"] > 0
            else None
        ),

        kmer_weight=(
            config["kmer_weight"]
        ),

        # ----------------------------------------------------
        # 5' GC constraint
        # ----------------------------------------------------

        five_prime_window=(
            FIVE_PRIME_WINDOW
            if config["five_prime_weight"] > 0
            else None
        ),

        five_prime_gc_min=(
            FIVE_PRIME_GC_MIN
            if config["five_prime_weight"] > 0
            else None
        ),

        five_prime_gc_max=(
            FIVE_PRIME_GC_MAX
            if config["five_prime_weight"] > 0
            else None
        ),

        five_prime_gc_weight=(
            config["five_prime_weight"]
        ),

        seed=SEED,
        return_history=False,
    )

    runtime = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------
    # Optimized sequence statistics
    # --------------------------------------------------------

    optimized_cai = calculate_cai(
        optimized,
        organism="ecoli",
    )

    optimized_gc = gc_content(
        optimized
    )

    # --------------------------------------------------------
    # Protein preservation
    #
    # IMPORTANT:
    # verify_translation() expects:
    #     original DNA
    #     optimized DNA
    #
    # Do NOT pass original_protein here.
    # --------------------------------------------------------

    verification = verify_translation(
        sequence,
        optimized,
    )

    protein_preserved = (
        verification["valid"]
    )

    # --------------------------------------------------------
    # Sequence changes
    # --------------------------------------------------------

    changed = changed_positions(
        sequence,
        optimized,
    )

    codons = total_codons(
        sequence
    )

    edit = edit_rate(
        sequence,
        optimized,
    )

    # --------------------------------------------------------
    # K-mer statistics
    # --------------------------------------------------------

    max_kmer, max_kmer_frequency = (
        get_max_kmer_stats(
            optimized,
            KMER_SIZE,
        )
    )

    # --------------------------------------------------------
    # 5' GC statistics
    # --------------------------------------------------------

    five_prime_gc = get_five_prime_gc(
        optimized,
        FIVE_PRIME_WINDOW,
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "gene": gene_name,

        "experiment": config["name"],

        "type": config["type"],

        "kmer_size": (
            KMER_SIZE
            if config["kmer_weight"] > 0
            else ""
        ),

        "kmer_threshold": (
            config["kmer_threshold"]
            if config["kmer_threshold"] is not None
            else ""
        ),

        "kmer_weight": (
            config["kmer_weight"]
        ),

        "five_prime_window": (
            FIVE_PRIME_WINDOW
            if config["five_prime_weight"] > 0
            else ""
        ),

        "five_prime_gc_min": (
            FIVE_PRIME_GC_MIN
            if config["five_prime_weight"] > 0
            else ""
        ),

        "five_prime_gc_max": (
            FIVE_PRIME_GC_MAX
            if config["five_prime_weight"] > 0
            else ""
        ),

        "five_prime_gc_weight": (
            config["five_prime_weight"]
        ),

        "original_cai": (
            original_cai
        ),

        "optimized_cai": (
            optimized_cai
        ),

        "delta_cai": (
            optimized_cai
            - original_cai
        ),

        "original_gc_percent": (
            original_gc * 100
        ),

        "optimized_gc_percent": (
            optimized_gc * 100
        ),

        "five_prime_gc_percent": (
            five_prime_gc * 100
            if config["five_prime_weight"] > 0
            else ""
        ),

        "max_kmer": (
            max_kmer
            if config["kmer_weight"] > 0
            else ""
        ),

        "max_kmer_frequency": (
            max_kmer_frequency
            if config["kmer_weight"] > 0
            else ""
        ),

        "changed_codons": (
            changed
        ),

        "total_codons": (
            codons
        ),

        "edit_rate_percent": (
            edit * 100
        ),

        "runtime_seconds": (
            runtime
        ),

        "protein_preserved": (
            protein_preserved
        ),

        "sequence_changed": (
            sequence != optimized
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    experiments = build_experiments()

    total_runs = (
        len(GENES)
        * len(experiments)
    )

    print()
    print("=" * 75)
    print("CONSTRAINT SENSITIVITY EXPERIMENT")
    print("=" * 75)

    print(
        f"Genes         : {len(GENES)}"
    )

    print(
        f"Configurations: {len(experiments)}"
    )

    print(
        f"Total runs    : {total_runs}"
    )

    print(
        "Host          : ecoli"
    )

    print(
        f"K-mer size    : {KMER_SIZE}"
    )

    print(
        f"5' GC window  : "
        f"{FIVE_PRIME_WINDOW} bp"
    )

    print(
        f"5' GC range   : "
        f"{FIVE_PRIME_GC_MIN * 100:.0f}% - "
        f"{FIVE_PRIME_GC_MAX * 100:.0f}%"
    )

    print("=" * 75)
    print()

    results = []

    successful = 0
    errors = 0
    protein_failures = 0

    run_number = 0

    # ========================================================
    # Loop through genes
    # ========================================================

    for gene_name, filename in GENES.items():

        fasta_path = (
            FASTA_DIR / filename
        )

        if not fasta_path.exists():

            print(
                f"ERROR: Missing FASTA file: "
                f"{fasta_path}"
            )

            errors += len(experiments)

            continue

        sequence = read_fasta(
            fasta_path
        )

        print("-" * 75)

        print(
            f"Gene: {gene_name}"
        )

        print(
            f"Original length: "
            f"{len(sequence)} bp"
        )

        print(
            f"Original GC: "
            f"{gc_content(sequence) * 100:.2f}%"
        )

        print("-" * 75)

        # ====================================================
        # Run all configurations
        # ====================================================

        for config in experiments:

            run_number += 1

            print()

            print(
                f"[{run_number}/{total_runs}] "
                f"{config['name']}"
            )

            try:

                result = run_one(
                    gene_name,
                    sequence,
                    config,
                )

                results.append(
                    result
                )

                successful += 1

                if not result[
                    "protein_preserved"
                ]:
                    protein_failures += 1

                print(
                    f"  CAI: "
                    f"{result['original_cai']:.4f} "
                    f"-> "
                    f"{result['optimized_cai']:.4f}"
                )

                print(
                    f"  Delta CAI: "
                    f"{result['delta_cai']:+.4f}"
                )

                print(
                    f"  GC: "
                    f"{result['original_gc_percent']:.2f}% "
                    f"-> "
                    f"{result['optimized_gc_percent']:.2f}%"
                )

                if (
                    result[
                        "five_prime_gc_percent"
                    ]
                    != ""
                ):

                    print(
                        f"  5' GC: "
                        f"{result['five_prime_gc_percent']:.2f}%"
                    )

                if (
                    result["max_kmer"]
                    != ""
                ):

                    print(
                        f"  Max 6-mer: "
                        f"{result['max_kmer']} "
                        f"("
                        f"{result['max_kmer_frequency']:.4f}"
                        f")"
                    )

                print(
                    f"  Edit rate: "
                    f"{result['edit_rate_percent']:.2f}%"
                )

                print(
                    f"  Runtime: "
                    f"{result['runtime_seconds']:.4f} s"
                )

                print(
                    f"  Protein preserved: "
                    f"{result['protein_preserved']}"
                )

            except Exception as exc:

                errors += 1

                print(
                    f"  ERROR: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "gene",
        "experiment",
        "type",

        "kmer_size",
        "kmer_threshold",
        "kmer_weight",

        "five_prime_window",
        "five_prime_gc_min",
        "five_prime_gc_max",
        "five_prime_gc_weight",

        "original_cai",
        "optimized_cai",
        "delta_cai",

        "original_gc_percent",
        "optimized_gc_percent",

        "five_prime_gc_percent",

        "max_kmer",
        "max_kmer_frequency",

        "changed_codons",
        "total_codons",
        "edit_rate_percent",

        "runtime_seconds",

        "protein_preserved",
        "sequence_changed",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print("=" * 75)
    print("CONSTRAINT SENSITIVITY COMPLETE")
    print("=" * 75)

    print(
        f"Successful runs          : "
        f"{successful}/{total_runs}"
    )

    print(
        f"Errors                   : "
        f"{errors}"
    )

    print(
        f"Protein preservation fail: "
        f"{protein_failures}"
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()