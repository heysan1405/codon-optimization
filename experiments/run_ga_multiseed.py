from pathlib import Path
import csv
import sys
import time
import statistics

# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.sequence import gc_content
from core.cai import calculate_cai
from core.verification import translate_dna
from optimizers.genetic_algorithm import optimize_ga


FASTA_DIR = PROJECT_ROOT / "data" / "genes" / "fasta"
OUTPUT_DIR = PROJECT_ROOT / "results" / "csv"

RESULT_FILE = OUTPUT_DIR / "ga_multiseed.csv"
SUMMARY_FILE = OUTPUT_DIR / "ga_multiseed_summary.csv"


# ============================================================
# Dataset
# ============================================================

GENES = {
    "GFP": "gene01_gfp.fasta",
    "Firefly luciferase": "gene02_luc.fasta",
    "HBB": "gene03_hbb.fasta",
    "INS": "gene04_ins.fasta",
    "LYZ": "gene05_lyz.fasta",
    "ACTB": "gene06_actb.fasta",
    "TP53": "gene07_tp53.fasta",
    "lacZ": "gene08_lacz.fasta",
    "groL": "gene09_grol.fasta",
    "sodA": "gene10_soda.fasta",
}


# Five independent random seeds
SEEDS = [42, 123, 456, 789, 2026]


# ============================================================
# FASTA
# ============================================================

def read_fasta(path):

    sequence_parts = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if not line.startswith(">"):
                sequence_parts.append(line)

    return "".join(sequence_parts).upper()


# ============================================================
# Validation
# ============================================================

def validate_dna(sequence):

    if not sequence:
        raise ValueError("Empty DNA sequence.")

    invalid = set(sequence) - set("ATGC")

    if invalid:
        raise ValueError(
            f"Invalid DNA bases: {sorted(invalid)}"
        )

    if len(sequence) % 3 != 0:
        raise ValueError(
            f"DNA length {len(sequence)} is not divisible by 3."
        )


# ============================================================
# Edit rate
# ============================================================

def calculate_edit_rate(original, optimized):

    original_codons = [
        original[i:i + 3]
        for i in range(0, len(original), 3)
    ]

    optimized_codons = [
        optimized[i:i + 3]
        for i in range(0, len(optimized), 3)
    ]

    changed = sum(
        a != b
        for a, b in zip(
            original_codons,
            optimized_codons
        )
    )

    total = len(original_codons)

    if total == 0:
        return 0.0, 0

    return changed / total, changed


# ============================================================
# Main experiment
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    detailed_results = []

    print("=" * 80)
    print("MULTI-SEED GA ROBUSTNESS EXPERIMENT")
    print("=" * 80)

    print(f"\nGenes: {len(GENES)}")
    print(f"Seeds: {SEEDS}")
    print(f"Total GA runs: {len(GENES) * len(SEEDS)}")

    print("\nGA configuration:")
    print("  Population: 100")
    print("  Generations: 100")
    print("  Mutation rate: 0.02")
    print("  Crossover rate: 0.80")
    print("  Tournament size: 3")
    print("  Elitism: 2")
    print("  Host: E. coli")

    # ========================================================
    # Run every gene with every seed
    # ========================================================

    for gene, filename in GENES.items():

        dna_path = FASTA_DIR / filename

        if not dna_path.exists():

            print(
                f"\nERROR: Missing file for {gene}: "
                f"{dna_path}"
            )

            continue

        original_dna = read_fasta(
            dna_path
        )

        validate_dna(
            original_dna
        )

        original_cai = calculate_cai(
            original_dna,
            organism="ecoli"
        )

        original_gc = (
            gc_content(original_dna) * 100
        )

        original_protein = translate_dna(
            original_dna
        )

        print("\n" + "-" * 80)
        print(f"GENE: {gene}")
        print(
            f"Original CAI: {original_cai:.4f}"
        )
        print(
            f"Original GC: {original_gc:.2f}%"
        )

        for seed in SEEDS:

            print(
                f"\n  Running seed {seed}..."
            )

            start_time = time.perf_counter()

            try:

                result = optimize_ga(
                    dna_sequence=original_dna,
                    organism="ecoli",

                    population_size=100,
                    generations=100,

                    mutation_rate=0.02,
                    crossover_rate=0.80,

                    tournament_size=3,
                    elitism=2,

                    cai_weight=1.0,

                    gc_weight=10.0,

                    local_gc_weight=5.0,
                    repeat_weight=5.0,
                    homopolymer_weight=5.0,
                    restriction_weight=5.0,

                    edit_weight=0.0,

                    seed=seed,

                    return_history=True,
                )

                runtime = (
                    time.perf_counter()
                    - start_time
                )

                optimized_dna = result[
                    "sequence"
                ]

                # ------------------------------------------------
                # Metrics
                # ------------------------------------------------

                optimized_cai = calculate_cai(
                    optimized_dna,
                    organism="ecoli"
                )

                optimized_gc = (
                    gc_content(
                        optimized_dna
                    ) * 100
                )

                cai_change = (
                    optimized_cai
                    - original_cai
                )

                cai_improvement_percent = (
                    cai_change
                    / original_cai
                    * 100
                    if original_cai != 0
                    else 0.0
                )

                edit_rate, changed_codons = (
                    calculate_edit_rate(
                        original_dna,
                        optimized_dna
                    )
                )

                optimized_protein = (
                    translate_dna(
                        optimized_dna
                    )
                )

                protein_preserved = (
                    original_protein
                    == optimized_protein
                )

                # ------------------------------------------------
                # Result
                # ------------------------------------------------

                row = {

                    "gene": gene,

                    "seed": seed,

                    "original_cai":
                        round(
                            original_cai,
                            6
                        ),

                    "optimized_cai":
                        round(
                            optimized_cai,
                            6
                        ),

                    "cai_change":
                        round(
                            cai_change,
                            6
                        ),

                    "cai_improvement_percent":
                        round(
                            cai_improvement_percent,
                            4
                        ),

                    "original_gc_percent":
                        round(
                            original_gc,
                            4
                        ),

                    "optimized_gc_percent":
                        round(
                            optimized_gc,
                            4
                        ),

                    "gc_change":
                        round(
                            optimized_gc
                            - original_gc,
                            4
                        ),

                    "changed_codons":
                        changed_codons,

                    "edit_rate":
                        round(
                            edit_rate,
                            6
                        ),

                    "protein_preserved":
                        protein_preserved,

                    "runtime_seconds":
                        round(
                            runtime,
                            4
                        ),
                }

                detailed_results.append(
                    row
                )

                print(
                    f"    CAI: "
                    f"{original_cai:.4f}"
                    f" -> "
                    f"{optimized_cai:.4f}"
                )

                print(
                    f"    CAI improvement: "
                    f"{cai_improvement_percent:.2f}%"
                )

                print(
                    f"    GC: "
                    f"{original_gc:.2f}%"
                    f" -> "
                    f"{optimized_gc:.2f}%"
                )

                print(
                    f"    Edit rate: "
                    f"{edit_rate * 100:.2f}%"
                )

                print(
                    f"    Protein preserved: "
                    f"{protein_preserved}"
                )

                print(
                    f"    Runtime: "
                    f"{runtime:.2f}s"
                )

            except Exception as e:

                print(
                    f"    ERROR: {e}"
                )


    # ========================================================
    # Save detailed results
    # ========================================================

    if not detailed_results:

        print(
            "\nNo results generated."
        )

        return

    detail_fields = list(
        detailed_results[0].keys()
    )

    with open(
        RESULT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=detail_fields
        )

        writer.writeheader()
        writer.writerows(
            detailed_results
        )


    # ========================================================
    # Calculate per-gene statistics
    # ========================================================

    summary_results = []

    for gene in GENES:

        gene_results = [
            row
            for row in detailed_results
            if row["gene"] == gene
        ]

        if not gene_results:
            continue

        cai_values = [
            row["optimized_cai"]
            for row in gene_results
        ]

        cai_changes = [
            row["cai_change"]
            for row in gene_results
        ]

        cai_improvements = [
            row["cai_improvement_percent"]
            for row in gene_results
        ]

        edit_rates = [
            row["edit_rate"]
            for row in gene_results
        ]

        runtimes = [
            row["runtime_seconds"]
            for row in gene_results
        ]

        optimized_gc_values = [
            row["optimized_gc_percent"]
            for row in gene_results
        ]

        protein_status = all(
            row["protein_preserved"]
            for row in gene_results
        )

        summary_results.append({

            "gene": gene,

            "number_of_seeds":
                len(gene_results),

            "original_cai":
                gene_results[0][
                    "original_cai"
                ],

            "mean_optimized_cai":
                round(
                    statistics.mean(
                        cai_values
                    ),
                    6
                ),

            "std_optimized_cai":
                round(
                    statistics.stdev(
                        cai_values
                    )
                    if len(cai_values) > 1
                    else 0.0,
                    6
                ),

            "mean_cai_change":
                round(
                    statistics.mean(
                        cai_changes
                    ),
                    6
                ),

            "std_cai_change":
                round(
                    statistics.stdev(
                        cai_changes
                    )
                    if len(cai_changes) > 1
                    else 0.0,
                    6
                ),

            "mean_cai_improvement_percent":
                round(
                    statistics.mean(
                        cai_improvements
                    ),
                    4
                ),

            "min_cai_improvement_percent":
                round(
                    min(cai_improvements),
                    4
                ),

            "max_cai_improvement_percent":
                round(
                    max(cai_improvements),
                    4
                ),

            "mean_optimized_gc_percent":
                round(
                    statistics.mean(
                        optimized_gc_values
                    ),
                    4
                ),

            "std_optimized_gc_percent":
                round(
                    statistics.stdev(
                        optimized_gc_values
                    )
                    if len(optimized_gc_values) > 1
                    else 0.0,
                    4
                ),

            "mean_edit_rate":
                round(
                    statistics.mean(
                        edit_rates
                    ),
                    6
                ),

            "std_edit_rate":
                round(
                    statistics.stdev(
                        edit_rates
                    )
                    if len(edit_rates) > 1
                    else 0.0,
                    6
                ),

            "mean_runtime_seconds":
                round(
                    statistics.mean(
                        runtimes
                    ),
                    4
                ),

            "protein_preserved_all_seeds":
                protein_status,
        })


    # ========================================================
    # Save summary
    # ========================================================

    summary_fields = list(
        summary_results[0].keys()
    )

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields
        )

        writer.writeheader()
        writer.writerows(
            summary_results
        )


    # ========================================================
    # Print summary
    # ========================================================

    print("\n")
    print("=" * 100)
    print("MULTI-SEED SUMMARY")
    print("=" * 100)

    print(
        f"{'Gene':<24}"
        f"{'Mean CAI':>12}"
        f"{'SD CAI':>10}"
        f"{'Mean ΔCAI':>12}"
        f"{'Mean Edit':>12}"
        f"{'Protein':>12}"
    )

    print("-" * 100)

    for row in summary_results:

        print(
            f"{row['gene']:<24}"
            f"{row['mean_optimized_cai']:>12.4f}"
            f"{row['std_optimized_cai']:>10.4f}"
            f"{row['mean_cai_change']:>12.4f}"
            f"{row['mean_edit_rate'] * 100:>11.2f}%"
            f"{str(row['protein_preserved_all_seeds']):>12}"
        )

    print("-" * 100)

    print(
        f"\nDetailed results:"
        f"\n{RESULT_FILE}"
    )

    print(
        f"\nStatistical summary:"
        f"\n{SUMMARY_FILE}"
    )

    print(
        f"\nTotal successful runs: "
        f"{len(detailed_results)}"
        f"/{len(GENES) * len(SEEDS)}"
    )

    failed_proteins = [
        row
        for row in detailed_results
        if not row["protein_preserved"]
    ]

    if failed_proteins:

        print(
            "\nWARNING: Protein preservation "
            "failed in some runs."
        )

        for row in failed_proteins:

            print(
                f"  {row['gene']} "
                f"(seed={row['seed']})"
            )

    else:

        print(
            "\nProtein preservation: "
            "PASSED for every run."
        )


if __name__ == "__main__":
    main()