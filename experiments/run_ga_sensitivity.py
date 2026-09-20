import sys
import os
import csv
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from optimizers.genetic_algorithm import optimize_ga


SEQUENCE = "ATGGCCGCG"
ORGANISM = "ecoli"

OUTPUT_FILE = "results/csv/ga_sensitivity.csv"


EXPERIMENTS = [
    # Population size
    {
        "parameter": "population_size",
        "value": 20,
        "population_size": 20,
        "generations": 50,
        "mutation_rate": 0.02,
        "crossover_rate": 0.8,
    },
    {
        "parameter": "population_size",
        "value": 50,
        "population_size": 50,
        "generations": 50,
        "mutation_rate": 0.02,
        "crossover_rate": 0.8,
    },
    {
        "parameter": "population_size",
        "value": 100,
        "population_size": 100,
        "generations": 50,
        "mutation_rate": 0.02,
        "crossover_rate": 0.8,
    },

    # Generations
    {
        "parameter": "generations",
        "value": 25,
        "population_size": 50,
        "generations": 25,
        "mutation_rate": 0.02,
        "crossover_rate": 0.8,
    },
    {
        "parameter": "generations",
        "value": 50,
        "population_size": 50,
        "generations": 50,
        "mutation_rate": 0.02,
        "crossover_rate": 0.8,
    },
    {
        "parameter": "generations",
        "value": 100,
        "population_size": 50,
        "generations": 100,
        "mutation_rate": 0.02,
        "crossover_rate": 0.8,
    },

    # Mutation rate
    {
        "parameter": "mutation_rate",
        "value": 0.01,
        "population_size": 50,
        "generations": 50,
        "mutation_rate": 0.01,
        "crossover_rate": 0.8,
    },
    {
        "parameter": "mutation_rate",
        "value": 0.02,
        "population_size": 50,
        "generations": 50,
        "mutation_rate": 0.02,
        "crossover_rate": 0.8,
    },
    {
        "parameter": "mutation_rate",
        "value": 0.05,
        "population_size": 50,
        "generations": 50,
        "mutation_rate": 0.05,
        "crossover_rate": 0.8,
    },

    # Crossover rate
    {
        "parameter": "crossover_rate",
        "value": 0.6,
        "population_size": 50,
        "generations": 50,
        "mutation_rate": 0.02,
        "crossover_rate": 0.6,
    },
    {
        "parameter": "crossover_rate",
        "value": 0.8,
        "population_size": 50,
        "generations": 50,
        "mutation_rate": 0.02,
        "crossover_rate": 0.8,
    },
    {
        "parameter": "crossover_rate",
        "value": 1.0,
        "population_size": 50,
        "generations": 50,
        "mutation_rate": 0.02,
        "crossover_rate": 1.0,
    },
]


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    rows = []

    for experiment_id, config in enumerate(EXPERIMENTS, start=1):

        print(
            f"Running {experiment_id}/{len(EXPERIMENTS)}: "
            f"{config['parameter']}={config['value']}"
        )

        start_time = time.perf_counter()

        result = optimize_ga(
            SEQUENCE,
            organism=ORGANISM,
            population_size=config["population_size"],
            generations=config["generations"],
            mutation_rate=config["mutation_rate"],
            crossover_rate=config["crossover_rate"],
            seed=42,
            return_history=True,
        )

        runtime = time.perf_counter() - start_time

        history = result["history"]
        final = history[-1]

        rows.append({
            "experiment_id": experiment_id,
            "parameter": config["parameter"],
            "value": config["value"],
            "population_size": config["population_size"],
            "generations": config["generations"],
            "mutation_rate": config["mutation_rate"],
            "crossover_rate": config["crossover_rate"],
            "optimized_sequence": result["sequence"],
            "best_fitness": final["best_fitness"],
            "best_cai": final["best_cai"],
            "best_gc": final["best_gc"],
            "edit_rate": final["edit_rate"],
            "runtime_seconds": runtime,
            "seed": 42,
        })

        print(f"  Sequence: {result['sequence']}")
        print(f"  Fitness:  {final['best_fitness']}")
        print(f"  CAI:      {final['best_cai']}")
        print(f"  GC:       {final['best_gc']}")
        print(f"  Runtime:  {runtime:.6f} s")
        print()

    fieldnames = [
        "experiment_id",
        "parameter",
        "value",
        "population_size",
        "generations",
        "mutation_rate",
        "crossover_rate",
        "optimized_sequence",
        "best_fitness",
        "best_cai",
        "best_gc",
        "edit_rate",
        "runtime_seconds",
        "seed",
    ]

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 60)
    print(f"Sensitivity experiment complete.")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Experiments: {len(rows)}")


if __name__ == "__main__":
    main()