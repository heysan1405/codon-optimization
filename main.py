"""
Main entry point for the codon optimization project.

This module connects the sequence utilities, optimizers,
constraints, and verification modules.
"""

from core.sequence import (
    clean_dna,
    gc_content
)

from core.cai import (
    calculate_cai
)

from core.verification import (
    verify_translation
)

from constraints.gc import (
    gc_in_range
)

from constraints.local_gc import (
    local_gc_in_range,
    local_gc_violations
)

from constraints.repeats import (
    find_repeats
)

from constraints.homopolymer import (
    find_homopolymers
)

from constraints.restriction_sites import (
    find_restriction_sites
)

from optimizers.greedy import (
    optimize_greedy
)

from optimizers.milp import (
    optimize_milp
)

from optimizers.genetic_algorithm import (
    optimize_ga
)

from optimizers.dual_host import (
    optimize_dual_host
)


def optimize(
    sequence,
    organism,
    method="greedy",
    gc_min=None,
    gc_max=None,
    local_gc_window=None,
    local_gc_min=None,
    local_gc_max=None,
    repeat_length=None,
    max_homopolymer=5,
    forbidden_sites=None,
    population_size=50,
    generations=100,
    mutation_rate=0.05,
    crossover_rate=0.8,
    seed=None,
    organism2=None,
    weight1=0.5,
    weight2=0.5,
    edit_weight=0.0
):
    """
    Main optimization interface.

    Parameters
    ----------
    sequence : str
        Input DNA sequence.

    organism : str
        Primary target organism.

    method : str
        Optimization method:
        greedy, milp, ga, or dual_host.

    gc_min : float
        Minimum global GC fraction.

    gc_max : float
        Maximum global GC fraction.

    local_gc_window : int
        Sliding-window size for local GC analysis.

    local_gc_min : float
        Minimum local GC fraction.

    local_gc_max : float
        Maximum local GC fraction.

    repeat_length : int
        Minimum repeated sequence length.

    max_homopolymer : int
        Maximum allowed homopolymer length.

    forbidden_sites : list
        Restriction sites to avoid.

    population_size : int
        GA population size.

    generations : int
        Number of GA generations.

    mutation_rate : float
        GA mutation rate.

    crossover_rate : float
        GA crossover rate.

    seed : int
        Random seed for reproducibility.

    organism2 : str
        Second organism for dual-host optimization.

    weight1 : float
        Weight of organism 1 in dual-host optimization.

    weight2 : float
        Weight of organism 2 in dual-host optimization.

    edit_weight : float
        Penalty for changing codons from the original sequence.
        Used by the genetic algorithm.
    """

    # --------------------------------------------------
    # Input preparation
    # --------------------------------------------------

    sequence = clean_dna(sequence)

    if forbidden_sites is None:
        forbidden_sites = []

    method = method.lower()

    # --------------------------------------------------
    # Select optimizer
    # --------------------------------------------------

    if method == "greedy":

        optimized_sequence = optimize_greedy(
            sequence,
            organism
        )

    elif method == "milp":

        optimized_sequence = optimize_milp(
            sequence=sequence,
            organism=organism,
            gc_min=gc_min,
            gc_max=gc_max
        )

    elif method == "ga":

        optimized_sequence = optimize_ga(
            dna_sequence=sequence,
            organism=organism,
            gc_min=gc_min,
            gc_max=gc_max,
            local_gc_window=local_gc_window,
            local_gc_min=local_gc_min,
            local_gc_max=local_gc_max,
            repeat_length=repeat_length,
            max_homopolymer=max_homopolymer,
            forbidden_sites=forbidden_sites,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            edit_weight=edit_weight,
            seed=seed
        )

    elif method == "dual_host":

        if organism2 is None:
            raise ValueError(
                "organism2 is required for dual_host optimization."
            )

        optimized_sequence = optimize_dual_host(
            sequence=sequence,
            organism1=organism,
            organism2=organism2,
            weight1=weight1,
            weight2=weight2
        )

    else:

        raise ValueError(
            f"Unknown optimization method: {method}. "
            "Choose greedy, milp, ga, or dual_host."
        )

    # --------------------------------------------------
    # Protein verification
    # --------------------------------------------------

    translation_result = verify_translation(
        sequence,
        optimized_sequence
    )

    # --------------------------------------------------
    # Global GC analysis
    # --------------------------------------------------

    original_gc = gc_content(
        sequence
    )

    optimized_gc = gc_content(
        optimized_sequence
    )

    global_gc_valid = gc_in_range(
        optimized_sequence,
        gc_min,
        gc_max
    )

    # --------------------------------------------------
    # Local GC analysis
    # --------------------------------------------------

    local_gc_valid = None
    local_gc_violation_list = []

    if local_gc_window is not None:

        local_gc_valid = local_gc_in_range(
            optimized_sequence,
            local_gc_window,
            local_gc_min,
            local_gc_max
        )

        local_gc_violation_list = (
            local_gc_violations(
                optimized_sequence,
                local_gc_window,
                local_gc_min,
                local_gc_max
            )
        )

    # --------------------------------------------------
    # Repeat analysis
    # --------------------------------------------------

    repeat_results = []

    if repeat_length is not None:

        repeat_results = find_repeats(
            optimized_sequence,
            repeat_length
        )

    # --------------------------------------------------
    # Homopolymer analysis
    # --------------------------------------------------

    homopolymer_results = find_homopolymers(
        optimized_sequence,
        max_homopolymer
    )

    # --------------------------------------------------
    # Restriction-site analysis
    # --------------------------------------------------

    restriction_results = find_restriction_sites(
        optimized_sequence,
        forbidden_sites
    )

    # --------------------------------------------------
    # CAI analysis
    # --------------------------------------------------

    original_cai = calculate_cai(
        sequence,
        organism
    )

    optimized_cai = calculate_cai(
        optimized_sequence,
        organism
    )

    # --------------------------------------------------
    # Codon edit analysis
    # --------------------------------------------------

    original_codons = [
        sequence[i:i + 3]
        for i in range(0, len(sequence), 3)
    ]

    optimized_codons = [
        optimized_sequence[i:i + 3]
        for i in range(0, len(optimized_sequence), 3)
    ]

    changed_codons = sum(
        original != optimized
        for original, optimized
        in zip(
            original_codons,
            optimized_codons
        )
    )

    total_codons = len(original_codons)

    edit_rate = (
        changed_codons / total_codons
        if total_codons > 0
        else 0.0
    )

    # --------------------------------------------------
    # Final validation
    # --------------------------------------------------

    all_constraints_valid = (
        global_gc_valid
        and (
            local_gc_valid
            if local_gc_valid is not None
            else True
        )
        and len(repeat_results) == 0
        and len(homopolymer_results) == 0
        and len(restriction_results) == 0
    )

    # --------------------------------------------------
    # Result object
    # --------------------------------------------------

    result = {
        "original_sequence": sequence,

        "optimized_sequence": optimized_sequence,

        "method": method,

        "organism": organism,

        "original_length": len(sequence),

        "optimized_length": len(
            optimized_sequence
        ),

        "original_gc": original_gc,

        "optimized_gc": optimized_gc,

        "original_gc_percentage": (
            original_gc * 100
        ),

        "optimized_gc_percentage": (
            optimized_gc * 100
        ),

        "original_cai": original_cai,

        "optimized_cai": optimized_cai,

        "protein_preserved": (
            translation_result["valid"]
        ),

        "original_protein": (
            translation_result[
                "original_protein"
            ]
        ),

        "optimized_protein": (
            translation_result[
                "optimized_protein"
            ]
        ),

        "global_gc_valid": global_gc_valid,

        "local_gc_valid": local_gc_valid,

        "local_gc_violations": (
            local_gc_violation_list
        ),

        "repeat_violations": (
            repeat_results
        ),

        "homopolymer_violations": (
            homopolymer_results
        ),

        "restriction_site_violations": (
            restriction_results
        ),

        "changed_codons": changed_codons,

        "total_codons": total_codons,

        "edit_rate": edit_rate,

        "edit_weight": edit_weight,

        "all_constraints_valid": (
            all_constraints_valid
        )
    }

    return result


def print_result(result):
    """
    Print an optimization result in a readable format.
    """

    print("\n" + "=" * 60)
    print("CODON OPTIMIZATION RESULT")
    print("=" * 60)

    print(
        f"\nMethod       : {result['method']}"
    )

    print(
        f"Organism     : {result['organism']}"
    )

    print(
        f"\nOriginal DNA :\n"
        f"{result['original_sequence']}"
    )

    print(
        f"\nOptimized DNA:\n"
        f"{result['optimized_sequence']}"
    )

    print(
        f"\nOriginal GC  : "
        f"{result['original_gc_percentage']:.2f}%"
    )

    print(
        f"Optimized GC : "
        f"{result['optimized_gc_percentage']:.2f}%"
    )

    print(
        f"\nOriginal CAI : "
        f"{result['original_cai']:.4f}"
    )

    print(
        f"Optimized CAI: "
        f"{result['optimized_cai']:.4f}"
    )

    print(
        f"\nChanged codons : "
        f"{result['changed_codons']}/"
        f"{result['total_codons']}"
    )

    print(
        f"Edit rate      : "
        f"{result['edit_rate']:.2%}"
    )

    print(
        f"Edit weight    : "
        f"{result['edit_weight']}"
    )

    print(
        f"\nProtein preserved: "
        f"{result['protein_preserved']}"
    )

    print(
        f"Global GC valid  : "
        f"{result['global_gc_valid']}"
    )

    print(
        f"Local GC valid   : "
        f"{result['local_gc_valid']}"
    )

    print(
        f"Repeat violations: "
        f"{len(result['repeat_violations'])}"
    )

    print(
        f"Homopolymer violations: "
        f"{len(result['homopolymer_violations'])}"
    )

    print(
        f"Restriction-site violations: "
        f"{len(result['restriction_site_violations'])}"
    )

    print(
        f"\nAll constraints valid: "
        f"{result['all_constraints_valid']}"
    )

    print("=" * 60)


def main():
    """
    Example command-line execution.
    """

    sequence = "ATGGCCGCG"

    result = optimize(
        sequence=sequence,
        organism="ecoli",
        method="ga",
        gc_min=0.40,
        gc_max=0.70,
        population_size=20,
        generations=10,
        mutation_rate=0.05,
        crossover_rate=0.8,
        seed=42,
        edit_weight=0.0
    )

    print_result(result)


if __name__ == "__main__":
    main()