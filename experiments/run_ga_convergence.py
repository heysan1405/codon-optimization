import csv
import os

from optimizers.genetic_algorithm import optimize_ga


SEQUENCE = "ATGGCCGCG"
ORGANISM = "ecoli"

POPULATION_SIZE = 20
GENERATIONS = 50
SEED = 42

OUTPUT_FILE = "results/csv/ga_convergence.csv"


def main():

    result = optimize_ga(
        SEQUENCE,
        organism=ORGANISM,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        seed=SEED,
        return_history=True,
    )

    history = result["history"]

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    fieldnames = [
        "generation",
        "best_fitness",
        "mean_fitness",
        "best_cai",
        "best_gc",
        "edit_rate",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(history)

    print(
        f"Optimized sequence: {result['sequence']}"
    )

    print(
        f"History entries: {len(history)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()