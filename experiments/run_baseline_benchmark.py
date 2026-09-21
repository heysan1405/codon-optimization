from pathlib import Path
import csv
import sys
import time

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

RESULT_FILE = OUTPUT_DIR / "ga_benchmark.csv"
HISTORY_FILE = OUTPUT_DIR / "ga_convergence_real_genes.csv"


# ============================================================
# Biological benchmark dataset
# ============================================================

GENES = {
    "GFP": {
        "dna": "gene01_gfp.fasta",
        "protein": "gene01_gfp_protein.fasta",
    },
    "Firefly luciferase": {
        "dna": "gene02_luc.fasta",
        "protein": "gene02_luc_protein.fasta",
    },
    "HBB": {
        "dna": "gene03_hbb.fasta",
        "protein": "gene03_hbb_protein.fasta",
    },
    "INS": {
        "dna": "gene04_ins.fasta",
        "protein": "gene04_ins_protein.fasta",
    },
    "LYZ": {
        "dna": "gene05_lyz.fasta",
        "protein": "gene05_lyz_protein.fasta",
    },
    "ACTB": {
        "dna": "gene06_actb.fasta",
        "protein": "gene06_actb_protein.fasta",
    },
    "TP53": {
        "dna": "gene07_tp53.fasta",
        "protein": "gene07_tp53_protein.fasta",
    },
    "lacZ": {
        "dna": "gene08_lacz.fasta",
        "protein": "gene08_lacz_protein.fasta",
    },
    "groL": {
        "dna": "gene09_grol.fasta",
        "protein": "gene09_grol_protein.fasta",
    },
    "sodA": {
        "dna": "gene10_soda.fasta",
        "protein": "gene10_soda_protein.fasta",
    },
}


# ============================================================
# FASTA reader
# ============================================================

def read_fasta(path):

    header = ""
    sequence_parts = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                header = line[1:]
            else:
                sequence_parts.append(line)

    sequence = "".join(sequence_parts).upper()

    return header, sequence


# ============================================================
# Basic validation
# ============================================================

def validate_dna(sequence):

    if not sequence:
        raise ValueError("DNA sequence is empty.")

    invalid = set(sequence) - set("ATGC")

    if invalid:
        raise ValueError(
            f"Invalid DNA bases found: {sorted(invalid)}"
        )

    if len(sequence) % 3 != 0:
        raise ValueError(
            f"DNA length {len(sequence)} is not divisible by 3."
        )


# ============================================================
# Codon statistics
# ============================================================

def codon_statistics(sequence):

    codons = [
        sequence[i:i + 3]
        for i in range(0, len(sequence), 3)
    ]

    return {
        "codon_count": len(codons),
        "unique_codons": len(set(codons)),
    }


# ============================================================
# Homopolymer statistics
# ============================================================

def homopolymer_metrics(sequence):

    longest = 1
    current = 1
    count_ge4 = 0

    for i in range(1, len(sequence)):

        if sequence[i] == sequence[i - 1]:

            current += 1

        else:

            if current >= 4:
                count_ge4 += 1

            longest = max(longest, current)
            current = 1

    if current >= 4:
        count_ge4 += 1

    longest = max(longest, current)

    return longest, count_ge4


# ============================================================
# Repeated 6-mer statistics
# ============================================================

def repeat_count(sequence, repeat_length=6):

    counts = {}

    for i in range(
        len(sequence) - repeat_length + 1
    ):

        kmer = sequence[
            i:i + repeat_length
        ]

        counts[kmer] = counts.get(kmer, 0) + 1

    repeated = 0

    for count in counts.values():

        if count > 1:
            repeated += count - 1

    return repeated


# ============================================================
# Edit statistics
# ============================================================

def codon_edit_rate(original, optimized):

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
# Calculate sequence metrics
# ============================================================

def sequence_metrics(sequence):

    validate_dna(sequence)

    codon_stats = codon_statistics(sequence)

    longest_homopolymer, homopolymer_count = (
        homopolymer_metrics(sequence)
    )

    repeats = repeat_count(
        sequence,
        repeat_length=6
    )

    cai = calculate_cai(
        sequence,
        organism="ecoli"
    )

    return {
        "dna_length": len(sequence),
        "codon_count": codon_stats["codon_count"],
        "unique_codons": codon_stats["unique_codons"],
        "gc_percent": gc_content(sequence) * 100,
        "cai_ecoli": cai,
        "longest_homopolymer": longest_homopolymer,
        "homopolymer_count_ge4": homopolymer_count,
        "repeat_count_6mer": repeats,
    }


# ============================================================
# Main experiment
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    benchmark_results = []
    convergence_results = []

    print("=" * 75)
    print("GA BIOLOGICAL BENCHMARK")
    print("=" * 75)

    print("\nGA configuration:")
    print("  Host: E. coli")
    print("  Population: 100")
    print("  Generations: 100")
    print("  Mutation rate: 0.02")
    print("  Crossover rate: 0.80")
    print("  Tournament size: 3")
    print("  Elitism: 2")
    print("  Random seed: 42")

    print("\nStarting optimization...\n")

    for gene, files in GENES.items():

        print("-" * 75)
        print(f"Processing: {gene}")

        dna_path = FASTA_DIR / files["dna"]
        protein_path = FASTA_DIR / files["protein"]

        if not dna_path.exists():
            print(f"ERROR: Missing DNA file: {dna_path}")
            continue

        if not protein_path.exists():
            print(f"ERROR: Missing protein file: {protein_path}")
            continue

        try:

            _, original_dna = read_fasta(
                dna_path
            )

            _, reference_protein = read_fasta(
                protein_path
            )

            validate_dna(original_dna)

            # ------------------------------------------------
            # Original metrics
            # ------------------------------------------------

            original_metrics = sequence_metrics(
                original_dna
            )

            # ------------------------------------------------
            # Run GA
            # ------------------------------------------------

            start_time = time.perf_counter()

            ga_result = optimize_ga(
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

                seed=42,

                return_history=True,
            )

            runtime = (
                time.perf_counter()
                - start_time
            )

            optimized_dna = ga_result["sequence"]
            history = ga_result["history"]

            # ------------------------------------------------
            # Optimized metrics
            # ------------------------------------------------

            optimized_metrics = sequence_metrics(
                optimized_dna
            )

            # ------------------------------------------------
            # Protein verification
            # ------------------------------------------------

            original_translated = translate_dna(
                original_dna
            )

            optimized_translated = translate_dna(
                optimized_dna
            )

            protein_preserved = (
                original_translated
                == optimized_translated
            )

            reference_matches_original = (
                original_translated
                == reference_protein
            )

            # ------------------------------------------------
            # Edit statistics
            # ------------------------------------------------

            edit_rate, changed_codons = (
                codon_edit_rate(
                    original_dna,
                    optimized_dna
                )
            )

            # ------------------------------------------------
            # Record benchmark result
            # ------------------------------------------------

            row = {

                "gene": gene,

                "dna_length": len(original_dna),

                "original_gc_percent": round(
                    original_metrics["gc_percent"],
                    4
                ),

                "optimized_gc_percent": round(
                    optimized_metrics["gc_percent"],
                    4
                ),

                "original_cai": round(
                    original_metrics["cai_ecoli"],
                    6
                ),

                "optimized_cai": round(
                    optimized_metrics["cai_ecoli"],
                    6
                ),

                "cai_change": round(
                    optimized_metrics["cai_ecoli"]
                    - original_metrics["cai_ecoli"],
                    6
                ),

                "original_longest_homopolymer":
                    original_metrics[
                        "longest_homopolymer"
                    ],

                "optimized_longest_homopolymer":
                    optimized_metrics[
                        "longest_homopolymer"
                    ],

                "original_homopolymer_count":
                    original_metrics[
                        "homopolymer_count_ge4"
                    ],

                "optimized_homopolymer_count":
                    optimized_metrics[
                        "homopolymer_count_ge4"
                    ],

                "original_repeat_6mer_count":
                    original_metrics[
                        "repeat_count_6mer"
                    ],

                "optimized_repeat_6mer_count":
                    optimized_metrics[
                        "repeat_count_6mer"
                    ],

                "changed_codons": changed_codons,

                "edit_rate": round(
                    edit_rate,
                    6
                ),

                "protein_preserved":
                    protein_preserved,

                "reference_protein_matches":
                    reference_matches_original,

                "runtime_seconds": round(
                    runtime,
                    4
                ),

                "generations":
                    len(history) - 1,

            }

            benchmark_results.append(row)

            # ------------------------------------------------
            # Save convergence history
            # ------------------------------------------------

            for entry in history:

                convergence_results.append({

                    "gene": gene,

                    "generation":
                        entry["generation"],

                    "best_fitness":
                        entry["best_fitness"],

                    "mean_fitness":
                        entry["mean_fitness"],

                    "best_cai":
                        entry["best_cai"],

                    "best_gc_percent":
                        entry["best_gc"] * 100,

                    "edit_rate":
                        entry["edit_rate"],

                })

            # ------------------------------------------------
            # Print result
            # ------------------------------------------------

            print(
                f"Original CAI: "
                f"{original_metrics['cai_ecoli']:.4f}"
            )

            print(
                f"Optimized CAI: "
                f"{optimized_metrics['cai_ecoli']:.4f}"
            )

            print(
                f"Original GC: "
                f"{original_metrics['gc_percent']:.2f}%"
            )

            print(
                f"Optimized GC: "
                f"{optimized_metrics['gc_percent']:.2f}%"
            )

            print(
                f"Changed codons: "
                f"{changed_codons}"
            )

            print(
                f"Edit rate: "
                f"{edit_rate:.4f}"
            )

            print(
                f"Protein preserved: "
                f"{protein_preserved}"
            )

            print(
                f"Reference protein matches: "
                f"{reference_matches_original}"
            )

            print(
                f"Runtime: "
                f"{runtime:.2f} seconds"
            )

        except Exception as e:

            print(
                f"ERROR processing {gene}: {e}"
            )

            import traceback
            traceback.print_exc()

    # ========================================================
    # Save benchmark CSV
    # ========================================================

    if benchmark_results:

        fieldnames = list(
            benchmark_results[0].keys()
        )

        with open(
            RESULT_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(
                benchmark_results
            )

    # ========================================================
    # Save convergence CSV
    # ========================================================

    if convergence_results:

        fieldnames = list(
            convergence_results[0].keys()
        )

        with open(
            HISTORY_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(
                convergence_results
            )

    # ========================================================
    # Final summary
    # ========================================================

    print("\n")
    print("=" * 75)
    print("FINAL GA SUMMARY")
    print("=" * 75)

    print(
        f"{'Gene':<24}"
        f"{'Orig CAI':>10}"
        f"{'Opt CAI':>10}"
        f"{'Edit %':>10}"
        f"{'Protein':>12}"
    )

    print("-" * 75)

    for row in benchmark_results:

        print(
            f"{row['gene']:<24}"
            f"{row['original_cai']:>10.4f}"
            f"{row['optimized_cai']:>10.4f}"
            f"{row['edit_rate'] * 100:>9.2f}%"
            f"{str(row['protein_preserved']):>12}"
        )

    print("-" * 75)

    print(
        f"\nBenchmark results:"
        f"\n{RESULT_FILE}"
    )

    print(
        f"\nConvergence results:"
        f"\n{HISTORY_FILE}"
    )

    print(
        f"\nSuccessful genes: "
        f"{len(benchmark_results)}/{len(GENES)}"
    )

    failed_proteins = [
        row["gene"]
        for row in benchmark_results
        if not row["protein_preserved"]
    ]

    if failed_proteins:

        print(
            "\nWARNING: Protein preservation failed for:"
        )

        for gene in failed_proteins:
            print(f"  - {gene}")

    else:

        print(
            "\nProtein preservation: "
            "PASSED for all optimized genes."
        )


if __name__ == "__main__":
    main()