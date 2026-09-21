"""
Benchmark the new GA constraints.

Experiments:
    1. Baseline GA
    2. GA + k-mer constraint
    3. GA + 5' GC constraint
    4. GA + k-mer + 5' GC

10 biological genes are tested.

Codon-pair bias is intentionally disabled until a real
E. coli codon-pair frequency table is available.
"""

import sys
import csv
import time
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from core.sequence import gc_content
from core.cai import calculate_cai
from core.similarity import edit_rate
from core.verification import (
    translate_dna,
    verify_translation,
)

from optimizers.genetic_algorithm import optimize_ga


# ============================================================
# CONFIGURATION
# ============================================================

ORGANISM = "ecoli"

SEED = 42

POPULATION_SIZE = 100

GENERATIONS = 100

MUTATION_RATE = 0.02

CROSSOVER_RATE = 0.80


# ============================================================
# NEW CONSTRAINT VALUES
# ============================================================

KMER_SIZE = 6

MAX_KMER_FREQUENCY = 0.02

KMER_WEIGHT = 5.0


FIVE_PRIME_WINDOW = 60

FIVE_PRIME_GC_MIN = 0.40

FIVE_PRIME_GC_MAX = 0.60

FIVE_PRIME_GC_WEIGHT = 5.0


# ============================================================
# PATHS
# ============================================================

DATA_DIR = (
    PROJECT_ROOT /
    "data" /
    "genes" /
    "fasta"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "results" /
    "csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "new_constraints_benchmark.csv"
)


# ============================================================
# GENES
# ============================================================

GENES = {

    "GFP":
        "gene01_gfp.fasta",

    "Firefly_luciferase":
        "gene02_luc.fasta",

    "HBB":
        "gene03_hbb.fasta",

    "INS":
        "gene04_ins.fasta",

    "LYZ":
        "gene05_lyz.fasta",

    "ACTB":
        "gene06_actb.fasta",

    "TP53":
        "gene07_tp53.fasta",

    "lacZ":
        "gene08_lacz.fasta",

    "groL":
        "gene09_grol.fasta",

    "sodA":
        "gene10_soda.fasta",
}


# ============================================================
# EXPERIMENTS
# ============================================================

EXPERIMENTS = {

    "Baseline_GA": {

        "kmer_size": None,

        "max_kmer_frequency": None,

        "kmer_weight": 0.0,

        "five_prime_window": None,

        "five_prime_gc_min": None,

        "five_prime_gc_max": None,

        "five_prime_gc_weight": 0.0,
    },

    "GA_kmer": {

        "kmer_size":
            KMER_SIZE,

        "max_kmer_frequency":
            MAX_KMER_FREQUENCY,

        "kmer_weight":
            KMER_WEIGHT,

        "five_prime_window":
            None,

        "five_prime_gc_min":
            None,

        "five_prime_gc_max":
            None,

        "five_prime_gc_weight":
            0.0,
    },

    "GA_5prime_GC": {

        "kmer_size":
            None,

        "max_kmer_frequency":
            None,

        "kmer_weight":
            0.0,

        "five_prime_window":
            FIVE_PRIME_WINDOW,

        "five_prime_gc_min":
            FIVE_PRIME_GC_MIN,

        "five_prime_gc_max":
            FIVE_PRIME_GC_MAX,

        "five_prime_gc_weight":
            FIVE_PRIME_GC_WEIGHT,
    },

    "GA_kmer_5prime_GC": {

        "kmer_size":
            KMER_SIZE,

        "max_kmer_frequency":
            MAX_KMER_FREQUENCY,

        "kmer_weight":
            KMER_WEIGHT,

        "five_prime_window":
            FIVE_PRIME_WINDOW,

        "five_prime_gc_min":
            FIVE_PRIME_GC_MIN,

        "five_prime_gc_max":
            FIVE_PRIME_GC_MAX,

        "five_prime_gc_weight":
            FIVE_PRIME_GC_WEIGHT,
    },
}


# ============================================================
# FASTA READER
# ============================================================

def read_fasta(path):

    sequence_parts = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as handle:

        for line in handle:

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                continue

            sequence_parts.append(
                line
            )

    sequence = "".join(
        sequence_parts
    ).upper()

    if not sequence:

        raise ValueError(
            f"Empty FASTA: {path}"
        )

    return sequence


# ============================================================
# GC
# ============================================================

def gc_percent(sequence):

    return gc_content(sequence) * 100


# ============================================================
# K-MER STATISTICS
# ============================================================

def calculate_kmer_stats(
    sequence,
    k
):

    if k is None or k <= 0:

        return {
            "most_frequent_kmer":
                "",

            "most_frequent_kmer_count":
                0,

            "most_frequent_kmer_frequency":
                0.0,
        }

    if len(sequence) < k:

        return {
            "most_frequent_kmer":
                "",

            "most_frequent_kmer_count":
                0,

            "most_frequent_kmer_frequency":
                0.0,
        }

    counts = {}

    total = (
        len(sequence)
        - k
        + 1
    )

    for i in range(total):

        kmer = sequence[
            i:i + k
        ]

        counts[kmer] = (
            counts.get(kmer, 0)
            + 1
        )

    most_common = max(
        counts,
        key=counts.get
    )

    count = counts[
        most_common
    ]

    return {

        "most_frequent_kmer":
            most_common,

        "most_frequent_kmer_count":
            count,

        "most_frequent_kmer_frequency":
            count / total,
    }


# ============================================================
# 5' GC
# ============================================================

def calculate_five_prime_gc(
    sequence,
    window
):

    if window is None:

        return None

    region = sequence[
        :window
    ]

    if not region:

        return None

    return gc_percent(
        region
    )


# ============================================================
# CHANGED CODONS
# ============================================================

def changed_codons(
    original,
    optimized
):

    original_codons = [
        original[i:i + 3]
        for i in range(
            0,
            len(original),
            3
        )
    ]

    optimized_codons = [
        optimized[i:i + 3]
        for i in range(
            0,
            len(optimized),
            3
        )
    ]

    return sum(
        a != b
        for a, b in zip(
            original_codons,
            optimized_codons
        )
    )


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    sequence,
    config
):

    start = time.perf_counter()

    optimized = optimize_ga(

        dna_sequence=
            sequence,

        organism=
            ORGANISM,

        population_size=
            POPULATION_SIZE,

        generations=
            GENERATIONS,

        mutation_rate=
            MUTATION_RATE,

        crossover_rate=
            CROSSOVER_RATE,

        tournament_size=3,

        elitism=2,

        cai_weight=1.0,

        gc_weight=10.0,

        target_gc_weight=10.0,

        local_gc_weight=5.0,

        repeat_weight=5.0,

        homopolymer_weight=5.0,

        restriction_weight=5.0,

        edit_weight=0.0,

        # ----------------------------------------------------
        # NEW CONSTRAINTS
        # ----------------------------------------------------

        codon_pair_table=None,

        codon_pair_weight=0.0,

        kmer_size=
            config[
                "kmer_size"
            ],

        max_kmer_frequency=
            config[
                "max_kmer_frequency"
            ],

        kmer_weight=
            config[
                "kmer_weight"
            ],

        five_prime_window=
            config[
                "five_prime_window"
            ],

        five_prime_gc_min=
            config[
                "five_prime_gc_min"
            ],

        five_prime_gc_max=
            config[
                "five_prime_gc_max"
            ],

        five_prime_gc_weight=
            config[
                "five_prime_gc_weight"
            ],

        seed=
            SEED,

        return_history=False,
    )

    runtime = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    verification = verify_translation(
        sequence,
        optimized
    )

    protein_preserved = (
        verification["valid"]
        and
        translate_dna(sequence)
        ==
        translate_dna(optimized)
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    original_cai = calculate_cai(
        sequence,
        ORGANISM
    )

    optimized_cai = calculate_cai(
        optimized,
        ORGANISM
    )

    original_gc = gc_percent(
        sequence
    )

    optimized_gc = gc_percent(
        optimized
    )

    five_prime_gc = (
        calculate_five_prime_gc(
            optimized,
            config[
                "five_prime_window"
            ]
        )
    )

    kmer_stats = (
        calculate_kmer_stats(
            optimized,
            config[
                "kmer_size"
            ]
        )
    )

    return {

        "original_cai":
            original_cai,

        "optimized_cai":
            optimized_cai,

        "delta_cai":
            optimized_cai -
            original_cai,

        "original_gc_percent":
            original_gc,

        "optimized_gc_percent":
            optimized_gc,

        "gc_change_percent":
            optimized_gc -
            original_gc,

        "five_prime_gc_percent":
            five_prime_gc,

        "kmer_size":
            config[
                "kmer_size"
            ],

        "max_kmer_frequency":
            config[
                "max_kmer_frequency"
            ],

        "most_frequent_kmer":
            kmer_stats[
                "most_frequent_kmer"
            ],

        "most_frequent_kmer_frequency":
            kmer_stats[
                "most_frequent_kmer_frequency"
            ],

        "edit_rate_percent":
            edit_rate(
                sequence,
                optimized
            ) * 100,

        "changed_codons":
            changed_codons(
                sequence,
                optimized
            ),

        "runtime_seconds":
            runtime,

        "protein_preserved":
            protein_preserved,

        "optimized_sequence":
            optimized,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_runs = (
        len(GENES)
        *
        len(EXPERIMENTS)
    )

    completed = 0

    rows = []

    print()

    print(
        "=" * 75
    )

    print(
        "NEW CONSTRAINT BENCHMARK"
    )

    print(
        "=" * 75
    )

    print(
        f"Genes       : {len(GENES)}"
    )

    print(
        f"Experiments : {len(EXPERIMENTS)}"
    )

    print(
        f"Total runs  : {total_runs}"
    )

    print(
        f"Host        : {ORGANISM}"
    )

    print()

    for gene_name, filename in GENES.items():

        fasta_path = (
            DATA_DIR /
            filename
        )

        sequence = read_fasta(
            fasta_path
        )

        print()

        print(
            "-" * 75
        )

        print(
            f"Gene: {gene_name}"
        )

        print(
            f"Original GC: "
            f"{gc_percent(sequence):.2f}%"
        )

        print(
            "-" * 75
        )

        for experiment_name, config in (
            EXPERIMENTS.items()
        ):

            completed += 1

            print()

            print(
                f"[{completed}/{total_runs}] "
                f"{experiment_name}"
            )

            try:

                metrics = run_experiment(
                    sequence,
                    config
                )

                rows.append({

                    "gene":
                        gene_name,

                    "experiment":
                        experiment_name,

                    **{
                        key: value

                        for key, value
                        in metrics.items()

                        if key !=
                        "optimized_sequence"
                    },

                    "error":
                        "",
                })

                print(
                    f"  CAI: "
                    f"{metrics['original_cai']:.4f}"
                    f" -> "
                    f"{metrics['optimized_cai']:.4f}"
                )

                print(
                    f"  Delta CAI: "
                    f"{metrics['delta_cai']:+.4f}"
                )

                print(
                    f"  GC: "
                    f"{metrics['optimized_gc_percent']:.2f}%"
                )

                if (
                    metrics[
                        "five_prime_gc_percent"
                    ]
                    is not None
                ):

                    print(
                        f"  5' GC: "
                        f"{metrics['five_prime_gc_percent']:.2f}%"
                    )

                if config[
                    "kmer_size"
                ] is not None:

                    print(
                        f"  Most frequent "
                        f"{config['kmer_size']}-mer: "
                        f"{metrics['most_frequent_kmer']} "
                        f"("
                        f"{metrics['most_frequent_kmer_frequency']:.4f}"
                        f")"
                    )

                print(
                    f"  Edit rate: "
                    f"{metrics['edit_rate_percent']:.2f}%"
                )

                print(
                    f"  Runtime: "
                    f"{metrics['runtime_seconds']:.4f} s"
                )

                print(
                    f"  Protein preserved: "
                    f"{metrics['protein_preserved']}"
                )

            except Exception as exc:

                print(
                    f"  ERROR: {exc}"
                )

                rows.append({

                    "gene":
                        gene_name,

                    "experiment":
                        experiment_name,

                    "error":
                        str(exc),
                })

    # ========================================================
    # CSV
    # ========================================================

    fieldnames = [

        "gene",

        "experiment",

        "original_cai",

        "optimized_cai",

        "delta_cai",

        "original_gc_percent",

        "optimized_gc_percent",

        "gc_change_percent",

        "five_prime_gc_percent",

        "kmer_size",

        "max_kmer_frequency",

        "most_frequent_kmer",

        "most_frequent_kmer_frequency",

        "edit_rate_percent",

        "changed_codons",

        "runtime_seconds",

        "protein_preserved",

        "error",
    ]

    for row in rows:

        for field in fieldnames:

            row.setdefault(
                field,
                ""
            )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    successful = [

        row

        for row in rows

        if not row.get(
            "error"
        )
    ]

    protein_failures = [

        row

        for row in successful

        if str(
            row.get(
                "protein_preserved"
            )
        ).lower()
        != "true"
    ]

    errors = [

        row

        for row in rows

        if row.get(
            "error"
        )
    ]

    print()

    print(
        "=" * 75
    )

    print(
        "NEW CONSTRAINT BENCHMARK COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"Successful runs          : "
        f"{len(successful)}/{total_runs}"
    )

    print(
        f"Errors                   : "
        f"{len(errors)}"
    )

    print(
        f"Protein preservation fail: "
        f"{len(protein_failures)}"
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":

    main()