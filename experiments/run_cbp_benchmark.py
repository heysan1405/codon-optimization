from pathlib import Path
import csv
import json

from optimizers.genetic_algorithm import optimize_ga
from core.cai import calculate_cai
from core.sequence import gc_content
from core.similarity import sequence_identity
from core.codon_pair import codon_pair_score

from constraints.homopolymer import find_homopolymers
from constraints.repeats import find_repeats
from constraints.restriction_sites import find_restriction_sites
# Restriction sites used by the benchmark
FORBIDDEN_RESTRICTION_SITES = [
    "GAATTC",  # EcoRI
    "GGATCC",  # BamHI
    "AAGCTT",  # HindIII
    "CTGCAG",  # PstI
    "GCGGCCGC" # NotI
]


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FASTA = (
    ROOT
    / "data"
    / "genes"
    / "fasta"
    / "gene01_gfp.fasta"
)

CBP_TABLE = (
    ROOT
    / "data"
    / "codon_pair"
    / "ecoli_codon_pair_table.json"
)

OUTPUT = (
    ROOT
    / "results"
    / "csv"
    / "cbp_benchmark.csv"
)


# ============================================================
# FASTA loader
# ============================================================

def load_fasta(path):
    lines = path.read_text().splitlines()

    sequence = "".join(
        line.strip()
        for line in lines
        if line.strip() and not line.startswith(">")
    )

    return sequence.upper()


# ============================================================
# Simple metrics
# ============================================================

def calculate_metrics(
    original,
    optimized,
    codon_pair_table,
):
    return {
        "original_length": len(original),
        "optimized_length": len(optimized),

        "cai": calculate_cai(
            optimized,
            "ecoli",
        ),

        "gc_percent": gc_content(
            optimized
        ) * 100,

        "similarity": sequence_identity(
            original,
            optimized,
        ),

        "codon_pair_score": codon_pair_score(
            optimized,
            codon_pair_table,
        ),

        "homopolymer_count": len(
            find_homopolymers(optimized)
        ),

        "repeat_count": len(
            find_repeats(optimized)
        ),

        "restriction_site_count": len(
            find_restriction_sites(optimized, FORBIDDEN_RESTRICTION_SITES)
        ),
    }


# ============================================================
# Main benchmark
# ============================================================

def main():

    print("=" * 70)
    print("CODON-PAIR BIAS BENCHMARK")
    print("=" * 70)

    original = load_fasta(INPUT_FASTA)

    with CBP_TABLE.open("r") as f:
        codon_pair_table_raw = json.load(f)

    codon_pair_table = {
        tuple(k.split("_")): float(v)
        for k, v in codon_pair_table_raw.items()
    }
    print()
    print("Input:")
    print(INPUT_FASTA)

    print()
    print("Sequence length:", len(original))
    print("Codon-pair entries:", len(codon_pair_table))

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    print()
    print("Running baseline GA...")

    baseline = optimize_ga(
        original,
        organism="ecoli",
        generations=30,
        population_size=30,
        seed=42,
        codon_pair_table=None,
        codon_pair_weight=0.0,
    )

    # --------------------------------------------------------
    # CBP
    # --------------------------------------------------------

    print()
    print("Running GA + codon-pair bias...")

    cbp = optimize_ga(
        original,
        organism="ecoli",
        generations=30,
        population_size=30,
        seed=42,
        codon_pair_table=codon_pair_table,
        codon_pair_weight=5.0,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    baseline_metrics = calculate_metrics(
        original,
        baseline,
        codon_pair_table,
    )

    cbp_metrics = calculate_metrics(
        original,
        cbp,
        codon_pair_table,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    rows = []

    for method, sequence, metrics in [
        (
            "baseline_ga",
            baseline,
            baseline_metrics,
        ),
        (
            "ga_with_codon_pair_bias",
            cbp,
            cbp_metrics,
        ),
    ]:

        row = {
            "method": method,
            "sequence": sequence,
        }

        row.update(metrics)

        rows.append(row)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print()

    for row in rows:

        print(
            f"{row['method']}:"
        )

        print(
            "  CAI:",
            row["cai"]
        )

        print(
            "  GC %:",
            row["gc_percent"]
        )

        print(
            "  Similarity:",
            row["similarity"]
        )

        print(
            "  Codon-pair score:",
            row["codon_pair_score"]
        )

        print(
            "  Homopolymers:",
            row["homopolymer_count"]
        )

        print(
            "  Repeats:",
            row["repeat_count"]
        )

        print(
            "  Restriction sites:",
            row["restriction_site_count"]
        )

        print()

    print("=" * 70)
    print("RESULT FILE")
    print("=" * 70)

    print(OUTPUT)


if __name__ == "__main__":
    main()
