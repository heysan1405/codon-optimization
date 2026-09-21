"""
Compare Greedy, MILP, and GA codon optimization
on the 10-gene biological benchmark.

Outputs:
    results/csv/optimizer_comparison.csv
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
from core.verification import translate_dna, verify_translation

from optimizers.greedy import optimize_greedy
from optimizers.milp import optimize_milp
from optimizers.genetic_algorithm import optimize_ga


# ============================================================
# CONFIGURATION
# ============================================================

ORGANISM = "ecoli"

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
    "optimizer_comparison.csv"
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
# BIOLOGICAL BENCHMARK
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
# CODON SPLITTER
# ============================================================

def get_codons(sequence):

    return [
        sequence[i:i + 3]
        for i in range(
            0,
            len(sequence),
            3
        )
    ]


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

    # verify_translation() returns a dictionary.
    # Extract only the Boolean validity result.

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

    # --------------------------------------------------------
    # CAI
    # --------------------------------------------------------

    original_cai = calculate_cai(
        original,
        ORGANISM
    )

    optimized_cai = calculate_cai(
        optimized,
        ORGANISM
    )

    delta_cai = (
        optimized_cai -
        original_cai
    )

    # --------------------------------------------------------
    # GC
    # --------------------------------------------------------

    original_gc = gc_content(
        original
    )

    optimized_gc = gc_content(
        optimized
    )

    original_gc_percent = (
        original_gc * 100
    )

    optimized_gc_percent = (
        optimized_gc * 100
    )

    gc_change_percent = (
        optimized_gc_percent -
        original_gc_percent
    )

    # --------------------------------------------------------
    # Codon changes
    # --------------------------------------------------------

    original_codons = get_codons(
        original
    )

    optimized_codons = get_codons(
        optimized
    )

    changed_codons = sum(
        original_codon != optimized_codon
        for original_codon,
        optimized_codon
        in zip(
            original_codons,
            optimized_codons
        )
    )

    total_codons = len(
        original_codons
    )

    # --------------------------------------------------------
    # Edit rate
    # --------------------------------------------------------

    edit_rate_percent = (
        edit_rate(
            original,
            optimized
        ) * 100
    )

    return {

        "original_length_bp":
            len(original),

        "optimized_length_bp":
            len(optimized),

        "original_cai":
            original_cai,

        "optimized_cai":
            optimized_cai,

        "delta_cai":
            delta_cai,

        "original_gc_percent":
            original_gc_percent,

        "optimized_gc_percent":
            optimized_gc_percent,

        "gc_change_percent":
            gc_change_percent,

        "edit_rate_percent":
            edit_rate_percent,

        "changed_codons":
            changed_codons,

        "total_codons":
            total_codons,

        "protein_length":
            len(original_protein),

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

    start_time = time.perf_counter()

    # ========================================================
    # GREEDY
    # ========================================================

    if optimizer_name == "Greedy":

        optimized = optimize_greedy(
            sequence,
            ORGANISM
        )

    # ========================================================
    # MILP
    # ========================================================

    elif optimizer_name == "MILP":

        optimized = optimize_milp(
            sequence,
            ORGANISM
        )

    # ========================================================
    # GA
    # ========================================================

    elif optimizer_name == "GA":

        # IMPORTANT:
        # optimize_ga() returns a STRING when
        # return_history=False.

        optimized = optimize_ga(

            dna_sequence=sequence,

            organism=ORGANISM,

            population_size=GA_POPULATION,

            generations=GA_GENERATIONS,

            mutation_rate=GA_MUTATION_RATE,

            crossover_rate=GA_CROSSOVER_RATE,

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

    # ========================================================
    # RUNTIME
    # ========================================================

    runtime = (
        time.perf_counter()
        - start_time
    )

    # ========================================================
    # METRICS
    # ========================================================

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

    completed_runs = 0

    rows = []

    # ========================================================
    # HEADER
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "CODON OPTIMIZER COMPARISON"
    )

    print(
        "=" * 75
    )

    print(
        f"Genes       : {len(GENES)}"
    )

    print(
        f"Optimizers  : {len(optimizers)}"
    )

    print(
        f"Total runs  : {total_runs}"
    )

    print(
        f"Target host : {ORGANISM}"
    )

    print(
        f"GA seed     : {GA_SEED}"
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
            f"File: {filename}"
        )

        print(
            "-" * 75
        )

        fasta_path = (
            DATA_DIR /
            filename
        )

        # ----------------------------------------------------
        # Check FASTA
        # ----------------------------------------------------

        if not fasta_path.exists():

            print(
                "ERROR: FASTA file not found:"
            )

            print(
                fasta_path
            )

            for optimizer_name in optimizers:

                rows.append({

                    "gene":
                        gene_name,

                    "optimizer":
                        optimizer_name,

                    "error":
                        "FASTA file not found"
                })

            continue

        # ----------------------------------------------------
        # Read FASTA
        # ----------------------------------------------------

        try:

            original = read_fasta(
                fasta_path
            )

        except Exception as exc:

            print(
                f"ERROR reading FASTA: {exc}"
            )

            for optimizer_name in optimizers:

                rows.append({

                    "gene":
                        gene_name,

                    "optimizer":
                        optimizer_name,

                    "error":
                        str(exc)
                })

            continue

        print(
            f"Original length: "
            f"{len(original)} bp"
        )

        # ----------------------------------------------------
        # Length validation
        # ----------------------------------------------------

        if len(original) % 3 != 0:

            print(
                "WARNING: sequence length "
                "is not divisible by 3."
            )

        # ====================================================
        # RUN OPTIMIZERS
        # ====================================================

        for optimizer_name in optimizers:

            completed_runs += 1

            print()

            print(
                f"[{completed_runs}/{total_runs}] "
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

                    **metrics
                })

                # ------------------------------------------------
                # Output
                # ------------------------------------------------

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
                    f"{metrics['original_gc_percent']:.2f}%"
                    f" -> "
                    f"{metrics['optimized_gc_percent']:.2f}%"
                )

                print(
                    f"  GC change: "
                    f"{metrics['gc_change_percent']:+.2f}%"
                )

                print(
                    f"  Changed codons: "
                    f"{metrics['changed_codons']}"
                    f"/"
                    f"{metrics['total_codons']}"
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

                    "optimizer":
                        optimizer_name,

                    "error":
                        str(exc)
                })

    # ========================================================
    # CSV OUTPUT
    # ========================================================

    fieldnames = [

        "gene",

        "optimizer",

        "original_length_bp",

        "optimized_length_bp",

        "original_cai",

        "optimized_cai",

        "delta_cai",

        "original_gc_percent",

        "optimized_gc_percent",

        "gc_change_percent",

        "edit_rate_percent",

        "changed_codons",

        "total_codons",

        "protein_length",

        "protein_preserved",

        "runtime_seconds",

        "error",
    ]

    # --------------------------------------------------------
    # Fill missing fields
    # --------------------------------------------------------

    for row in rows:

        for field in fieldnames:

            row.setdefault(
                field,
                ""
            )

    # --------------------------------------------------------
    # Write CSV
    # --------------------------------------------------------

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

    successful_rows = [

        row

        for row in rows

        if not row.get(
            "error"
        )
    ]

    protein_failures = [

        row

        for row in successful_rows

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
        "BENCHMARK COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"Successful runs            : "
        f"{len(successful_rows)}/{total_runs}"
    )

    print(
        f"Errors                     : "
        f"{len(errors)}"
    )

    print(
        f"Protein preservation fails : "
        f"{len(protein_failures)}"
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