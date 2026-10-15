import csv
import os
import statistics
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cai import calculate_cai
from core.sequence import get_codons, gc_content
from core.verification import verify_translation
from constraints.homopolymer import find_homopolymers
from optimizers.genetic_algorithm import optimize_ga, calculate_fitness


ORGANISM = "ecoli"
FASTA_PATH = PROJECT_ROOT / "data" / "genes" / "fasta" / "gene08_lacz.fasta"
OUTPUT_DIR = PROJECT_ROOT / "results" / "csv"
DETAIL_FILE = OUTPUT_DIR / "ga_sensitivity.csv"
SUMMARY_FILE = OUTPUT_DIR / "ga_sensitivity_summary.csv"

SEEDS = [42, 43, 44, 45, 46]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GC_MIN = 0.40
GC_MAX = 0.60
TARGET_GC = 0.55
MAX_HOMOPOLYMER = 6
FIVE_PRIME_WINDOW = 90
FIVE_PRIME_GC_MIN = 0.30
FIVE_PRIME_GC_MAX = 0.70

EXPERIMENTS = [
    {"parameter": "population_size", "value": 20, "population_size": 20, "generations": 50, "mutation_rate": 0.02, "crossover_rate": 0.8},
    {"parameter": "population_size", "value": 50, "population_size": 50, "generations": 50, "mutation_rate": 0.02, "crossover_rate": 0.8},
    {"parameter": "population_size", "value": 100, "population_size": 100, "generations": 50, "mutation_rate": 0.02, "crossover_rate": 0.8},
    {"parameter": "generations", "value": 25, "population_size": 50, "generations": 25, "mutation_rate": 0.02, "crossover_rate": 0.8},
    {"parameter": "generations", "value": 50, "population_size": 50, "generations": 50, "mutation_rate": 0.02, "crossover_rate": 0.8},
    {"parameter": "generations", "value": 100, "population_size": 50, "generations": 100, "mutation_rate": 0.02, "crossover_rate": 0.8},
    {"parameter": "mutation_rate", "value": 0.01, "population_size": 50, "generations": 50, "mutation_rate": 0.01, "crossover_rate": 0.8},
    {"parameter": "mutation_rate", "value": 0.02, "population_size": 50, "generations": 50, "mutation_rate": 0.02, "crossover_rate": 0.8},
    {"parameter": "mutation_rate", "value": 0.05, "population_size": 50, "generations": 50, "mutation_rate": 0.05, "crossover_rate": 0.8},
    {"parameter": "crossover_rate", "value": 0.6, "population_size": 50, "generations": 50, "mutation_rate": 0.02, "crossover_rate": 0.6},
    {"parameter": "crossover_rate", "value": 0.8, "population_size": 50, "generations": 50, "mutation_rate": 0.02, "crossover_rate": 0.8},
    {"parameter": "crossover_rate", "value": 1.0, "population_size": 50, "generations": 50, "mutation_rate": 0.02, "crossover_rate": 1.0},
]


def read_fasta(path):
    return "".join(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ).upper()


def evaluate_result(original, optimized):
    original_codons = get_codons(original)
    optimized_codons = get_codons(optimized)

    verification = verify_translation(original, optimized)
    gc = gc_content(optimized)
    five_prime = optimized[:FIVE_PRIME_WINDOW]
    five_prime_gc = gc_content(five_prime)
    homopolymer_violations = find_homopolymers(
        optimized, max_length=MAX_HOMOPOLYMER
    )

    fitness = calculate_fitness(
        optimized_codons,
        original_codons,
        ORGANISM,
        gc_min=GC_MIN,
        gc_max=GC_MAX,
        target_gc=TARGET_GC,
        max_homopolymer=MAX_HOMOPOLYMER,
        five_prime_window=FIVE_PRIME_WINDOW,
        five_prime_gc_min=FIVE_PRIME_GC_MIN,
        five_prime_gc_max=FIVE_PRIME_GC_MAX,
    )

    edits = sum(
        original_codon != optimized_codon
        for original_codon, optimized_codon
        in zip(original_codons, optimized_codons)
    )
    edit_fraction = edits / len(original_codons) if original_codons else 0.0

    gc_ok = GC_MIN <= gc <= GC_MAX
    five_prime_ok = FIVE_PRIME_GC_MIN <= five_prime_gc <= FIVE_PRIME_GC_MAX
    homopolymer_ok = not homopolymer_violations
    length_ok = len(original) == len(optimized)

    return {
        "best_fitness": fitness,
        "best_cai": calculate_cai(optimized, ORGANISM),
        "best_gc": gc,
        "five_prime_gc": five_prime_gc,
        "edit_rate": edit_fraction,
        "protein_preserved": verification["valid"],
        "length_preserved": length_ok,
        "gc_in_bounds": gc_ok,
        "five_prime_gc_in_bounds": five_prime_ok,
        "homopolymer_constraint_satisfied": homopolymer_ok,
        "homopolymer_violation_count": len(homopolymer_violations),
        "all_checks_passed": (
            verification["valid"] and length_ok and gc_ok
            and five_prime_ok and homopolymer_ok
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sequence = read_fasta(FASTA_PATH)

    if not sequence:
        raise ValueError(f"No DNA sequence found in {FASTA_PATH}")

    print("Gene: LACZ")
    print("Sequence length:", len(sequence))
    print("Device:", DEVICE.upper())
    print("Seeds per configuration:", len(SEEDS))
    print("Total runs:", len(EXPERIMENTS) * len(SEEDS))
    print("Constraints: GC 40%-60%, target GC 55%, maximum homopolymer 6, 5-prime GC 30%-70%")

    rows = []

    for exp_id, config in enumerate(EXPERIMENTS, start=1):
        for seed in SEEDS:
            label = f"{config['parameter']}={config['value']}"
            print(f"[{len(rows) + 1}/{len(EXPERIMENTS) * len(SEEDS)}] {label}, seed={seed}")

            start = time.perf_counter()
            result = optimize_ga(
                sequence,
                organism=ORGANISM,
                gc_min=GC_MIN,
                gc_max=GC_MAX,
                target_gc=TARGET_GC,
                max_homopolymer=MAX_HOMOPOLYMER,
                five_prime_window=FIVE_PRIME_WINDOW,
                five_prime_gc_min=FIVE_PRIME_GC_MIN,
                five_prime_gc_max=FIVE_PRIME_GC_MAX,
                population_size=config["population_size"],
                generations=config["generations"],
                mutation_rate=config["mutation_rate"],
                crossover_rate=config["crossover_rate"],
                seed=seed,
                return_history=True,
                device=DEVICE,
            )
            runtime = time.perf_counter() - start
            optimized = result["sequence"]
            metrics = evaluate_result(sequence, optimized)
            final_history = result["history"][-1]

            # Compare history with metrics recomputed from the returned sequence.
            history_consistent = (
                abs(final_history["best_fitness"] - metrics["best_fitness"]) < 1e-8
                and abs(final_history["best_cai"] - metrics["best_cai"]) < 1e-8
                and abs(final_history["best_gc"] - metrics["best_gc"]) < 1e-8
                and abs(final_history["edit_rate"] - metrics["edit_rate"]) < 1e-8
            )

            row = {
                "experiment_id": exp_id,
                "parameter": config["parameter"],
                "value": config["value"],
                "population_size": config["population_size"],
                "generations": config["generations"],
                "mutation_rate": config["mutation_rate"],
                "crossover_rate": config["crossover_rate"],
                "seed": seed,
                **metrics,
                "runtime_seconds": runtime,
                "history_matches_returned_sequence": history_consistent,
                "optimized_sequence": optimized,
            }
            rows.append(row)

            print(
                f"  fitness={metrics['best_fitness']:.6f} "
                f"CAI={metrics['best_cai']:.6f} "
                f"GC={metrics['best_gc']:.4f} "
                f"edit={metrics['edit_rate']:.4f} "
                f"checks={metrics['all_checks_passed']} "
                f"time={runtime:.2f}s "
                f"history_match={history_consistent}"
            )

    detail_fields = list(rows[0].keys())
    with DETAIL_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    groups = {}
    for row in rows:
        groups.setdefault((row["parameter"], row["value"]), []).append(row)

    for (parameter, value), group in groups.items():
        fitnesses = [r["best_fitness"] for r in group]
        cais = [r["best_cai"] for r in group]
        edit_rates = [r["edit_rate"] for r in group]
        runtimes = [r["runtime_seconds"] for r in group]

        summary_rows.append({
            "parameter": parameter,
            "value": value,
            "runs": len(group),
            "mean_fitness": statistics.mean(fitnesses),
            "std_fitness": statistics.stdev(fitnesses) if len(group) > 1 else 0.0,
            "best_fitness": max(fitnesses),
            "mean_cai": statistics.mean(cais),
            "std_cai": statistics.stdev(cais) if len(group) > 1 else 0.0,
            "mean_edit_rate": statistics.mean(edit_rates),
            "mean_runtime_seconds": statistics.mean(runtimes),
            "protein_preservation_rate": statistics.mean(
                float(r["protein_preserved"]) for r in group
            ),
            "constraint_pass_rate": statistics.mean(
                float(r["all_checks_passed"]) for r in group
            ),
            "history_consistency_rate": statistics.mean(
                float(r["history_matches_returned_sequence"]) for r in group
            ),
        })

    with SUMMARY_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\\nSensitivity experiment complete.")
    print("Detailed results:", DETAIL_FILE)
    print("Summary results:", SUMMARY_FILE)
    print("Runs completed:", len(rows))
    print("Protein-preservation rate:", sum(r["protein_preserved"] for r in rows), "/", len(rows))
    print("All-constraints pass rate:", sum(r["all_checks_passed"] for r in rows), "/", len(rows))
    print("History consistency:", sum(r["history_matches_returned_sequence"] for r in rows), "/", len(rows))


if __name__ == "__main__":
    main()
