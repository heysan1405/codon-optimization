"""
Controlled GC-target comparison for Greedy, MILP, and GA.

Target:
    GC = 50%

Allowed GC range:
    48% - 52%

Benchmark:
    10 biological genes

Output:
    results/csv/gc_target_comparison.csv
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

from optimizers.greedy import optimize_greedy
from optimizers.milp import optimize_milp
from optimizers.genetic_algorithm import optimize_ga


# ============================================================
# CONFIGURATION
# ============================================================

ORGANISM = "ecoli"

TARGET_GC = 0.50

GC_MIN = 0.48

GC_MAX = 0.52


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
    "gc_target_comparison.csv"
)


# ============================================================
# GA CONFIGURATION
# ============================================================

GA_SEED = 42

GA_POPULATION = 100

GA_GENERATIONS = 100

GA_MUTATION_RATE = 0.02

GA_CROSSOVER_RATE = 0.80


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

            sequence_parts.append(line)

    sequence = "".join(
        sequence_parts
    ).upper()

    if not sequence:

        raise ValueError(
            f"Empty FASTA sequence: {path}"
        )

    return sequence


# ============================================================
# GC METRIC
# ============================================================

def gc_percent(sequence):

    return gc_content(sequence) * 100


# ============================================================
# CODON DIFFERENCES
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
# METRICS
# ============================================================

def calculate_metrics(
    original,
    optimized
):

    original_protein = translate_dna(
        original
    )

    optimized_protein = translate_dna(
        optimized
    )

    verification = verify_translation(
        original,
        optimized
    )

    if isinstance(
        verification,
        dict
    ):

        verification_valid = (
            verification.get(
                "valid",
                False
            )
        )

    else:

        verification_valid = bool(
            verification
        )

    protein_preserved = (
        original_protein ==
        optimized_protein
        and
        verification_valid
    )

    original_cai = calculate_cai(
        original,
        ORGANISM
    )

    optimized_cai = calculate_cai(
        optimized,
        ORGANISM
    )

    original_gc = gc_percent(
        original
    )

    optimized_gc = gc_percent(
        optimized
    )

    target_deviation = abs(
        optimized_gc -
        (TARGET_GC * 100)
    )

    in_target_range = (
        optimized_gc >=
        GC_MIN * 100
        and
        optimized_gc <=
        GC_MAX * 100
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

        "target_gc_percent":
            TARGET_GC * 100,

        "gc_target_deviation":
            target_deviation,

        "within_target_range":
            in_target_range,

        "gc_change_percent":
            optimized_gc -
            original_gc,

        "edit_rate_percent":
            edit_rate(
                original,
                optimized
            ) * 100,

        "changed_codons":
            changed_codons(
                original,
                optimized
            ),

        "protein_preserved":
            protein_preserved,
    }


# ============================================================
# RUN OPTIMIZER
# ============================================================

def run_optimizer(
    optimizer_name,
    sequence
):

    start = time.perf_counter()

    # ========================================================
    # GREEDY
    # ========================================================

    if optimizer_name == "Greedy":

        """
        Greedy has no GC constraint in its current
        implementation.

        Therefore it is used here as the unconstrained
        baseline.
        """

        optimized = optimize_greedy(
            sequence,
            ORGANISM
        )

    # ========================================================
    # MILP
    # ========================================================

    elif optimizer_name == "MILP":

        """
        MILP supports global GC constraints.
        """

        optimized = optimize_milp(

            sequence=sequence,

            organism=ORGANISM,

            gc_min=GC_MIN,

            gc_max=GC_MAX
        )

    # ========================================================
    # GA
    # ========================================================

    elif optimizer_name == "GA":

        """
        GA uses the target GC objective.
        """

        optimized = optimize_ga(

            dna_sequence=sequence,

            organism=ORGANISM,

            target_gc=TARGET_GC,

            population_size=
                GA_POPULATION,

            generations=
                GA_GENERATIONS,

            mutation_rate=
                GA_MUTATION_RATE,

            crossover_rate=
                GA_CROSSOVER_RATE,

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

            seed=GA_SEED,

            return_history=False
        )

    else:

        raise ValueError(
            f"Unknown optimizer: "
            f"{optimizer_name}"
        )

    runtime = (
        time.perf_counter()
        - start
    )

    metrics = calculate_metrics(
        original=sequence,
        optimized=optimized
    )

    metrics[
        "runtime_seconds"
    ] = runtime

    return (
        optimized,
        metrics
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    optimizers = [
        "Greedy",
        "MILP",
        "GA"
    ]

    total_runs = (
        len(GENES) *
        len(optimizers)
    )

    completed = 0

    rows = []

    # ========================================================
    # HEADER
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "CONTROLLED GC-TARGET COMPARISON"
    )

    print(
        "=" * 75
    )

    print(
        f"Genes        : {len(GENES)}"
    )

    print(
        f"Optimizers   : {len(optimizers)}"
    )

    print(
        f"Total runs   : {total_runs}"
    )

    print(
        f"Target host  : {ORGANISM}"
    )

    print(
        f"Target GC    : {TARGET_GC * 100:.0f}%"
    )

    print(
        f"GC range     : "
        f"{GC_MIN * 100:.0f}% - "
        f"{GC_MAX * 100:.0f}%"
    )

    print()

    # ========================================================
    # PROCESS GENES
    # ========================================================

    for gene_name, filename in GENES.items():

        print()

        print(
            "-" * 75
        )

        print(
            f"Gene: {gene_name}"
        )

        print(
            "-" * 75
        )

        fasta_path = (
            DATA_DIR /
            filename
        )

        if not fasta_path.exists():

            print(
                f"ERROR: File not found: "
                f"{fasta_path}"
            )

            continue

        original = read_fasta(
            fasta_path
        )

        print(
            f"Original length: "
            f"{len(original)} bp"
        )

        print(
            f"Original GC: "
            f"{gc_percent(original):.2f}%"
        )

        # ====================================================
        # OPTIMIZERS
        # ====================================================

        for optimizer_name in optimizers:

            completed += 1

            print()

            print(
                f"[{completed}/{total_runs}] "
                f"{optimizer_name}"
            )

            try:

                optimized, metrics = (
                    run_optimizer(
                        optimizer_name,
                        original
                    )
                )

                rows.append({

                    "gene":
                        gene_name,

                    "optimizer":
                        optimizer_name,

                    "original_length_bp":
                        len(original),

                    **metrics
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

                print(
                    f"  Target deviation: "
                    f"{metrics['gc_target_deviation']:.2f}%"
                )

                print(
                    f"  Within 48-52%: "
                    f"{metrics['within_target_range']}"
                )

                print(
                    f"  Edit rate: "
                    f"{metrics['edit_rate_percent']:.2f}%"
                )

                print(
                    f"  Changed codons: "
                    f"{metrics['changed_codons']}"
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

                    "optimizer":
                        optimizer_name,

                    "error":
                        str(exc)
                })

    # ========================================================
    # WRITE CSV
    # ========================================================

    fieldnames = [

        "gene",

        "optimizer",

        "original_length_bp",

        "original_cai",

        "optimized_cai",

        "delta_cai",

        "original_gc_percent",

        "optimized_gc_percent",

        "target_gc_percent",

        "gc_target_deviation",

        "within_target_range",

        "gc_change_percent",

        "edit_rate_percent",

        "changed_codons",

        "protein_preserved",

        "runtime_seconds",

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

    gc_successes = [

        row

        for row in successful

        if row.get(
            "within_target_range"
        ) is True
    ]

    protein_failures = [

        row

        for row in successful

        if row.get(
            "protein_preserved"
        ) is not True
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
        "GC-TARGET BENCHMARK COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"Successful runs          : "
        f"{len(successful)}/{total_runs}"
    )

    print(
        f"Runs within 48-52% GC    : "
        f"{len(gc_successes)}/{len(successful)}"
    )

    print(
        f"Protein preservation fail: "
        f"{len(protein_failures)}"
    )

    print(
        f"Errors                   : "
        f"{len(errors)}"
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "=" * 75
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()