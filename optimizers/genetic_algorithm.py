"""
Genetic Algorithm for synonymous codon optimization.

Objectives:
    - Maximize CAI
    - Respect global GC constraints
    - Control local GC
    - Penalize repeats
    - Penalize homopolymers
    - Penalize forbidden restriction sites
    - Optionally minimize codon changes

The GA preserves the encoded protein by allowing only
synonymous codon substitutions.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from core.cai import calculate_cai
from core.sequence import get_codons, gc_content
from core.verification import (
    CODON_TO_AA,
    translate_dna,
    verify_translation,
)
from data.codon_tables import get_codon_table

from constraints.local_gc import local_gc_violations
from constraints.repeats import find_repeats
from constraints.homopolymer import find_homopolymers
from constraints.restriction_sites import find_restriction_sites


# ============================================================
# Helper functions
# ============================================================

def split_codons(sequence: str) -> List[str]:
    """Split DNA sequence into codons."""
    return get_codons(sequence)


def count_changes(
    original_codons: List[str],
    candidate_codons: List[str],
) -> int:
    """Count changed codon positions."""
    return sum(
        original != candidate
        for original, candidate in zip(
            original_codons,
            candidate_codons,
        )
    )


def edit_rate(
    original_codons: List[str],
    candidate_codons: List[str],
) -> float:
    """Calculate fraction of changed codons."""
    if not original_codons:
        return 0.0

    return count_changes(
        original_codons,
        candidate_codons,
    ) / len(original_codons)


# ============================================================
# Global GC
# ============================================================

def gc_penalty(
    gc: float,
    gc_min: Optional[float],
    gc_max: Optional[float],
) -> float:
    """Calculate global GC constraint violation."""

    penalty = 0.0

    if gc_min is not None and gc < gc_min:
        penalty += gc_min - gc

    if gc_max is not None and gc > gc_max:
        penalty += gc - gc_max

    return penalty


def is_gc_feasible(
    sequence: str,
    gc_min: Optional[float],
    gc_max: Optional[float],
) -> bool:
    """Check whether sequence satisfies global GC constraint."""

    gc = gc_content(sequence)

    if gc_min is not None and gc < gc_min:
        return False

    if gc_max is not None and gc > gc_max:
        return False

    return True


# ============================================================
# Local GC
# ============================================================

def local_gc_penalty(
    sequence: str,
    window_size: Optional[int],
    gc_min: Optional[float],
    gc_max: Optional[float],
) -> float:
    """
    Calculate local GC penalty.

    Each violating sliding window contributes a penalty
    proportional to how far it lies outside the requested range.
    """

    if window_size is None:
        return 0.0

    if gc_min is None and gc_max is None:
        return 0.0

    violations = local_gc_violations(
        sequence,
        window_size,
        gc_min,
        gc_max,
    )

    penalty = 0.0

    for window in violations:

        gc = window["gc_content"]

        if gc_min is not None and gc < gc_min:
            penalty += gc_min - gc

        if gc_max is not None and gc > gc_max:
            penalty += gc - gc_max

    return penalty


# ============================================================
# Repeat penalty
# ============================================================

def repeat_penalty(
    sequence: str,
    repeat_length: Optional[int],
) -> float:
    """Return the number of detected repeated fragments."""

    if repeat_length is None:
        return 0.0

    if repeat_length <= 0:
        return 0.0

    repeats = find_repeats(
        sequence,
        repeat_length,
    )

    return float(len(repeats))


# ============================================================
# Homopolymer penalty
# ============================================================

def homopolymer_penalty(
    sequence: str,
    max_homopolymer: Optional[int],
) -> float:
    """Return the number of homopolymer violations."""

    if max_homopolymer is None:
        return 0.0

    if max_homopolymer <= 0:
        return 0.0

    violations = find_homopolymers(
        sequence,
        max_homopolymer,
    )

    return float(len(violations))


# ============================================================
# Restriction-site penalty
# ============================================================

def restriction_penalty(
    sequence: str,
    forbidden_sites: Optional[List[str]],
) -> float:
    """Return the number of forbidden restriction-site occurrences."""

    if not forbidden_sites:
        return 0.0

    violations = find_restriction_sites(
        sequence,
        forbidden_sites,
    )

    return float(len(violations))


# ============================================================
# Synonymous codons
# ============================================================

def build_synonymous_codons(
    original_codons: List[str],
    codon_table: dict,
) -> List[List[str]]:
    """
    Build possible synonymous codons for every position.

    Stop codons remain unchanged.
    """

    choices_per_position = []

    for codon in original_codons:

        codon = codon.upper()

        if codon not in CODON_TO_AA:
            raise ValueError(
                f"Unknown codon: {codon}"
            )

        amino_acid = CODON_TO_AA[codon]

        # Preserve stop codons.
        if amino_acid == "*":
            choices_per_position.append([codon])
            continue

        if amino_acid not in codon_table:
            raise ValueError(
                f"Amino acid '{amino_acid}' "
                f"not found in codon table."
            )

        choices = list(
            codon_table[amino_acid].keys()
        )

        if not choices:
            raise ValueError(
                f"No synonymous codons available "
                f"for amino acid '{amino_acid}'."
            )

        choices_per_position.append(choices)

    return choices_per_position


# ============================================================
# Population initialization
# ============================================================

def initialize_population(
    original_codons: List[str],
    synonymous: List[List[str]],
    population_size: int,
    rng: random.Random,
) -> List[List[str]]:
    """Create initial GA population."""

    if population_size < 1:
        raise ValueError(
            "population_size must be at least 1."
        )

    population = [
        original_codons.copy()
    ]

    while len(population) < population_size:

        individual = []

        for position, choices in enumerate(
            synonymous
        ):

            original = original_codons[position]

            # Keep some original codons for diversity.
            if (
                original in choices
                and rng.random() < 0.30
            ):
                individual.append(original)

            else:
                individual.append(
                    rng.choice(choices)
                )

        population.append(individual)

    return population


# ============================================================
# Fitness
# ============================================================

def calculate_fitness(
    codons: List[str],
    original_codons: List[str],
    organism: str,
    gc_min: Optional[float] = None,
    gc_max: Optional[float] = None,

    local_gc_window: Optional[int] = None,
    local_gc_min: Optional[float] = None,
    local_gc_max: Optional[float] = None,

    repeat_length: Optional[int] = None,
    max_homopolymer: Optional[int] = None,
    forbidden_sites: Optional[List[str]] = None,

    cai_weight: float = 1.0,
    gc_weight: float = 10.0,
    local_gc_weight: float = 5.0,
    repeat_weight: float = 5.0,
    homopolymer_weight: float = 5.0,
    restriction_weight: float = 5.0,
    edit_weight: float = 0.0,
) -> float:
    """
    Calculate constraint-aware GA fitness.

    F =
        CAI
        - global GC penalty
        - local GC penalty
        - repeat penalty
        - homopolymer penalty
        - restriction-site penalty
        - edit-rate penalty
    """

    sequence = "".join(codons)

    # --------------------------------------------------------
    # CAI
    # --------------------------------------------------------

    cai = calculate_cai(
        sequence,
        organism,
    )

    # --------------------------------------------------------
    # Global GC
    # --------------------------------------------------------

    gc = gc_content(sequence)

    global_gc_violation = gc_penalty(
        gc,
        gc_min,
        gc_max,
    )

    # --------------------------------------------------------
    # Local GC
    # --------------------------------------------------------

    local_violation = local_gc_penalty(
        sequence,
        local_gc_window,
        local_gc_min,
        local_gc_max,
    )

    # --------------------------------------------------------
    # Repeats
    # --------------------------------------------------------

    repeat_violation = repeat_penalty(
        sequence,
        repeat_length,
    )

    # --------------------------------------------------------
    # Homopolymers
    # --------------------------------------------------------

    homopolymer_violation = homopolymer_penalty(
        sequence,
        max_homopolymer,
    )

    # --------------------------------------------------------
    # Restriction sites
    # --------------------------------------------------------

    restriction_violation = restriction_penalty(
        sequence,
        forbidden_sites,
    )

    # --------------------------------------------------------
    # Edit rate
    # --------------------------------------------------------

    rate = edit_rate(
        original_codons,
        codons,
    )

    # --------------------------------------------------------
    # Fitness
    # --------------------------------------------------------

    # Strong penalty for global GC violation.
    HARD_CONSTRAINT_PENALTY = 1000.0

    return (
        cai_weight * cai

        - gc_weight
        * HARD_CONSTRAINT_PENALTY
        * global_gc_violation

        - local_gc_weight
        * local_violation

        - repeat_weight
        * repeat_violation

        - homopolymer_weight
        * homopolymer_violation

        - restriction_weight
        * restriction_violation

        - edit_weight
        * rate
    )


# ============================================================
# Selection
# ============================================================

def tournament_selection(
    population: List[List[str]],
    fitnesses: List[float],
    rng: random.Random,
    tournament_size: int = 3,
) -> List[str]:
    """Select an individual using tournament selection."""

    size = min(
        tournament_size,
        len(population),
    )

    indices = rng.sample(
        range(len(population)),
        size,
    )

    winner = max(
        indices,
        key=lambda index: fitnesses[index],
    )

    return population[winner].copy()


# ============================================================
# Crossover
# ============================================================

def crossover(
    parent1: List[str],
    parent2: List[str],
    rng: random.Random,
) -> Tuple[List[str], List[str]]:
    """Single-point crossover."""

    if len(parent1) != len(parent2):
        raise ValueError(
            "Parents must have equal length."
        )

    if len(parent1) < 2:
        return (
            parent1.copy(),
            parent2.copy(),
        )

    point = rng.randint(
        1,
        len(parent1) - 1,
    )

    child1 = (
        parent1[:point]
        + parent2[point:]
    )

    child2 = (
        parent2[:point]
        + parent1[point:]
    )

    return child1, child2


# ============================================================
# Mutation
# ============================================================

def mutate(
    individual: List[str],
    synonymous: List[List[str]],
    mutation_rate: float,
    rng: random.Random,
) -> List[str]:
    """Replace codons with synonymous alternatives."""

    result = individual.copy()

    for position, choices in enumerate(
        synonymous
    ):

        if rng.random() >= mutation_rate:
            continue

        alternatives = [
            codon
            for codon in choices
            if codon != result[position]
        ]

        if alternatives:
            result[position] = rng.choice(
                alternatives
            )

    return result


# ============================================================
# Genetic Algorithm
# ============================================================

def optimize_ga(
    dna_sequence: str,
    organism: str = "ecoli",

    # Global GC
    gc_min: Optional[float] = None,
    gc_max: Optional[float] = None,

    # Local GC
    local_gc_window: Optional[int] = None,
    local_gc_min: Optional[float] = None,
    local_gc_max: Optional[float] = None,

    # Sequence constraints
    repeat_length: Optional[int] = None,
    max_homopolymer: Optional[int] = None,
    forbidden_sites: Optional[List[str]] = None,

    # GA parameters
    population_size: int = 100,
    generations: int = 100,
    mutation_rate: float = 0.02,
    crossover_rate: float = 0.80,
    tournament_size: int = 3,
    elitism: int = 2,

    # Fitness weights
    cai_weight: float = 1.0,
    gc_weight: float = 10.0,
    local_gc_weight: float = 5.0,
    repeat_weight: float = 5.0,
    homopolymer_weight: float = 5.0,
    restriction_weight: float = 5.0,
    edit_weight: float = 0.0,

    seed: Optional[int] = None,
) -> str:
    """Optimize DNA using a constraint-aware Genetic Algorithm."""

    # --------------------------------------------------------
    # Validate parameters
    # --------------------------------------------------------

    if population_size < 1:
        raise ValueError(
            "population_size must be at least 1."
        )

    if generations < 0:
        raise ValueError(
            "generations cannot be negative."
        )

    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(
            "mutation_rate must be between 0 and 1."
        )

    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError(
            "crossover_rate must be between 0 and 1."
        )

    if gc_min is not None and gc_max is not None:
        if gc_min > gc_max:
            raise ValueError(
                "gc_min cannot be greater than gc_max."
            )

    if (
        local_gc_min is not None
        and local_gc_max is not None
        and local_gc_min > local_gc_max
    ):
        raise ValueError(
            "local_gc_min cannot be greater "
            "than local_gc_max."
        )

    # --------------------------------------------------------
    # Prepare sequence
    # --------------------------------------------------------

    dna_sequence = dna_sequence.upper().strip()

    original_codons = split_codons(
        dna_sequence
    )

    original_protein = translate_dna(
        dna_sequence
    )

    codon_table = get_codon_table(
        organism
    )

    synonymous = build_synonymous_codons(
        original_codons,
        codon_table,
    )

    rng = random.Random(seed)

    # --------------------------------------------------------
    # Initial population
    # --------------------------------------------------------

    population = initialize_population(
        original_codons,
        synonymous,
        population_size,
        rng,
    )

    # --------------------------------------------------------
    # Fitness wrapper
    # --------------------------------------------------------

    def fitness(individual: List[str]) -> float:

        return calculate_fitness(
            individual,
            original_codons,
            organism,

            gc_min,
            gc_max,

            local_gc_window,
            local_gc_min,
            local_gc_max,

            repeat_length,
            max_homopolymer,
            forbidden_sites,

            cai_weight,
            gc_weight,
            local_gc_weight,
            repeat_weight,
            homopolymer_weight,
            restriction_weight,
            edit_weight,
        )

    # --------------------------------------------------------
    # Best solution
    # --------------------------------------------------------

    best_individual = original_codons.copy()

    best_fitness = fitness(
        best_individual
    )

    # --------------------------------------------------------
    # Evolution
    # --------------------------------------------------------

    for _generation in range(generations):

        fitnesses = [
            fitness(individual)
            for individual in population
        ]

        current_best_index = max(
            range(len(population)),
            key=lambda index: fitnesses[index],
        )

        current_best_fitness = fitnesses[
            current_best_index
        ]

        if current_best_fitness > best_fitness:

            best_fitness = current_best_fitness

            best_individual = population[
                current_best_index
            ].copy()

        # ----------------------------------------------------
        # Elitism
        # ----------------------------------------------------

        elite_count = min(
            max(elitism, 0),
            population_size,
        )

        ranked_indices = sorted(
            range(len(population)),
            key=lambda index: fitnesses[index],
            reverse=True,
        )

        new_population = [
            population[index].copy()
            for index in ranked_indices[:elite_count]
        ]

        # ----------------------------------------------------
        # Generate children
        # ----------------------------------------------------

        while len(new_population) < population_size:

            parent1 = tournament_selection(
                population,
                fitnesses,
                rng,
                tournament_size,
            )

            parent2 = tournament_selection(
                population,
                fitnesses,
                rng,
                tournament_size,
            )

            if rng.random() < crossover_rate:

                child1, child2 = crossover(
                    parent1,
                    parent2,
                    rng,
                )

            else:

                child1 = parent1.copy()
                child2 = parent2.copy()

            child1 = mutate(
                child1,
                synonymous,
                mutation_rate,
                rng,
            )

            child2 = mutate(
                child2,
                synonymous,
                mutation_rate,
                rng,
            )

            new_population.append(child1)

            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    # --------------------------------------------------------
    # Final population evaluation
    # --------------------------------------------------------

    final_fitnesses = [
        fitness(individual)
        for individual in population
    ]

    final_best_index = max(
        range(len(population)),
        key=lambda index: final_fitnesses[index],
    )

    if final_fitnesses[final_best_index] > best_fitness:

        best_individual = population[
            final_best_index
        ].copy()

    optimized_sequence = "".join(
        best_individual
    )

    # --------------------------------------------------------
    # Protein verification
    # --------------------------------------------------------

    verification_result = verify_translation(
        dna_sequence,
        optimized_sequence,
    )

    if not verification_result["valid"]:
        raise RuntimeError(
            "GA produced a sequence that does not "
            "preserve the original protein."
        )

    # --------------------------------------------------------
    # Global GC feasibility
    # --------------------------------------------------------

    if not is_gc_feasible(
        optimized_sequence,
        gc_min,
        gc_max,
    ):

        feasible_candidates = [
            individual
            for individual in population
            if is_gc_feasible(
                "".join(individual),
                gc_min,
                gc_max,
            )
        ]

        if feasible_candidates:

            best_feasible = max(
                feasible_candidates,
                key=fitness,
            )

            optimized_sequence = "".join(
                best_feasible
            )

        else:

            raise RuntimeError(
                "GA did not find a sequence satisfying "
                "the requested GC constraint."
            )

    # --------------------------------------------------------
    # Final protein verification
    # --------------------------------------------------------

    final_protein = translate_dna(
        optimized_sequence
    )

    if final_protein != original_protein:
        raise RuntimeError(
            "Protein sequence was not preserved."
        )

    return optimized_sequence


# ============================================================
# Compatibility alias
# ============================================================

def optimize_sequence_ga(
    dna_sequence: str,
    organism: str = "ecoli",
    **kwargs,
) -> str:
    """Compatibility wrapper."""

    return optimize_ga(
        dna_sequence,
        organism=organism,
        **kwargs,
    )