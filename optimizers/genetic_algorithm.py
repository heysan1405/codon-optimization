"""
Genetic Algorithm for synonymous codon optimization.

Features:
    - CAI optimization
    - Global GC constraints
    - Target GC optimization
    - Local GC constraints
    - Repeat constraints
    - Homopolymer constraints
    - Restriction-site constraints
    - Edit-rate penalty
    - Codon-pair bias
    - K-mer / sequence-complexity constraint
    - 5' local GC constraint
    - Reproducible random seed
    - GA convergence history
    - Protein preservation verification

The GA only uses synonymous codons, so the encoded
protein sequence must remain unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import random

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
# Utility functions
# ============================================================

def split_codons(sequence: str) -> List[str]:
    """Split a DNA sequence into codons."""

    sequence = sequence.upper().strip()

    if len(sequence) % 3 != 0:
        raise ValueError(
            "DNA sequence length must be divisible by 3."
        )

    return get_codons(sequence)


def count_changes(
    original_codons: List[str],
    candidate_codons: List[str],
) -> int:
    """Count codon positions that changed."""

    if len(original_codons) != len(candidate_codons):
        raise ValueError(
            "Codon sequences must have the same length."
        )

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
    """Calculate fraction of codons that changed."""

    if not original_codons:
        return 0.0

    return (
        count_changes(
            original_codons,
            candidate_codons,
        )
        / len(original_codons)
    )


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
    """Check global GC feasibility."""

    gc = gc_content(sequence)

    if gc_min is not None and gc < gc_min:
        return False

    if gc_max is not None and gc > gc_max:
        return False

    return True


# ============================================================
# Target GC
# ============================================================

def target_gc_deviation(
    sequence: str,
    target_gc: Optional[float],
) -> float:
    """Calculate deviation from target GC."""

    if target_gc is None:
        return 0.0

    if not 0.0 <= target_gc <= 1.0:
        raise ValueError(
            "target_gc must be between 0.0 and 1.0."
        )

    actual_gc = gc_content(sequence)

    return abs(actual_gc - target_gc)


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
    Calculate local GC violation penalty.
    """

    if window_size is None:
        return 0.0

    if window_size <= 0:
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

    for violation in violations:

        if isinstance(violation, dict):

            window_gc = violation.get(
                "gc_content",
                violation.get("gc"),
            )

            if window_gc is None:
                penalty += 1.0
                continue

        elif isinstance(violation, (tuple, list)):

            if len(violation) >= 3:
                window_gc = violation[2]
            else:
                penalty += 1.0
                continue

        else:
            penalty += 1.0
            continue

        if gc_min is not None and window_gc < gc_min:
            penalty += gc_min - window_gc

        if gc_max is not None and window_gc > gc_max:
            penalty += window_gc - gc_max

    return penalty


# ============================================================
# Repeats
# ============================================================

def repeat_penalty(
    sequence: str,
    repeat_length: Optional[int],
) -> float:
    """Calculate repeat penalty."""

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
# Homopolymers
# ============================================================

def homopolymer_penalty(
    sequence: str,
    max_homopolymer: Optional[int],
) -> float:
    """Calculate homopolymer penalty."""

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
# Restriction sites
# ============================================================

def restriction_penalty(
    sequence: str,
    forbidden_sites: Optional[List[str]],
) -> float:
    """Calculate restriction-site penalty."""

    if not forbidden_sites:
        return 0.0

    violations = find_restriction_sites(
        sequence,
        forbidden_sites,
    )

    return float(len(violations))


# ============================================================
# Codon-pair bias
# ============================================================

def codon_pair_penalty(
    sequence: str,
    codon_pair_table: Optional[dict] = None,
) -> float:
    """
    Calculate codon-pair bias penalty.

    The supplied table must contain experimentally observed
    codon-pair frequencies.

    Higher observed frequency = lower penalty.

    Missing pairs are ignored.

    IMPORTANT:
    This function does NOT derive codon-pair frequencies from
    single-codon frequencies.
    """

    if not codon_pair_table:
        return 0.0

    sequence = sequence.upper().strip()

    if len(sequence) % 3 != 0:
        raise ValueError(
            "DNA sequence length must be divisible by 3."
        )

    codons = [
        sequence[i:i + 3]
        for i in range(0, len(sequence), 3)
    ]

    if len(codons) < 2:
        return 0.0

    frequencies = []

    for i in range(len(codons) - 1):

        codon1 = codons[i]
        codon2 = codons[i + 1]

        pair = (
            codon1,
            codon2,
        )

        frequency = codon_pair_table.get(pair)

        # Also support:
        # {"AAA-AAG": frequency}
        if frequency is None:
            frequency = codon_pair_table.get(
                f"{codon1}-{codon2}"
            )

        if frequency is None:
            continue

        frequency = float(frequency)

        if frequency < 0:
            raise ValueError(
                f"Negative codon-pair frequency: {pair}"
            )

        frequencies.append(frequency)

    if not frequencies:
        return 0.0

    # Find the maximum observed frequency.
    all_frequencies = []

    for value in codon_pair_table.values():
        try:
            value = float(value)

            if value > 0:
                all_frequencies.append(value)

        except (TypeError, ValueError):
            continue

    if not all_frequencies:
        return 0.0

    max_frequency = max(
        all_frequencies
    )

    if max_frequency <= 0:
        return 0.0

    # Normalize observed frequencies to [0, 1].
    normalized_scores = [
        frequency / max_frequency
        for frequency in frequencies
    ]

    # High frequency -> low penalty.
    penalty = sum(
        1.0 - score
        for score in normalized_scores
    ) / len(normalized_scores)

    return penalty


# ============================================================
# K-mer / sequence complexity
# ============================================================

def kmer_frequency_penalty(
    sequence: str,
    kmer_size: Optional[int],
    max_kmer_frequency: Optional[float],
) -> float:
    """
    Penalize over-represented k-mers.

    max_kmer_frequency is the maximum allowed fraction
    of all k-mer positions occupied by one k-mer.

    Example:
        kmer_size = 6
        max_kmer_frequency = 0.02
    """

    if kmer_size is None:
        return 0.0

    if max_kmer_frequency is None:
        return 0.0

    if kmer_size <= 0:
        return 0.0

    if not 0.0 <= max_kmer_frequency <= 1.0:
        raise ValueError(
            "max_kmer_frequency must be between 0 and 1."
        )

    sequence = sequence.upper()

    if len(sequence) < kmer_size:
        return 0.0

    total = len(sequence) - kmer_size + 1

    counts: Dict[str, int] = {}

    for i in range(total):

        kmer = sequence[
            i:i + kmer_size
        ]

        counts[kmer] = (
            counts.get(kmer, 0) + 1
        )

    penalty = 0.0

    for count in counts.values():

        frequency = count / total

        if frequency > max_kmer_frequency:

            excess = (
                frequency
                - max_kmer_frequency
            )

            penalty += excess

    return penalty


# ============================================================
# 5' GC
# ============================================================

def five_prime_gc_penalty(
    sequence: str,
    window_size: Optional[int],
    gc_min: Optional[float],
    gc_max: Optional[float],
) -> float:
    """
    Calculate GC violation in the 5' region.

    Only the first window_size bases are evaluated.
    """

    if window_size is None:
        return 0.0

    if window_size <= 0:
        return 0.0

    if gc_min is None and gc_max is None:
        return 0.0

    sequence = sequence.upper()

    if not sequence:
        return 0.0

    window = sequence[
        :min(window_size, len(sequence))
    ]

    actual_gc = gc_content(window)

    penalty = 0.0

    if gc_min is not None and actual_gc < gc_min:

        penalty += (
            gc_min - actual_gc
        )

    if gc_max is not None and actual_gc > gc_max:

        penalty += (
            actual_gc - gc_max
        )

    return penalty


# ============================================================
# Synonymous codons
# ============================================================

def build_synonymous_codons(
    original_codons: List[str],
    codon_table: dict,
) -> List[List[str]]:
    """
    Build synonymous codon choices for every position.

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

            choices_per_position.append(
                [codon]
            )

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

        choices_per_position.append(
            choices
        )

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
    """Create the initial population."""

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

            if (
                original in choices
                and rng.random() < 0.30
            ):

                individual.append(
                    original
                )

            else:

                individual.append(
                    rng.choice(choices)
                )

        population.append(
            individual
        )

    return population


# ============================================================
# Fitness
# ============================================================

def calculate_fitness(
    codons: List[str],
    original_codons: List[str],
    organism: str,

    # Global GC
    gc_min: Optional[float] = None,
    gc_max: Optional[float] = None,

    # Target GC
    target_gc: Optional[float] = None,

    # Local GC
    local_gc_window: Optional[int] = None,
    local_gc_min: Optional[float] = None,
    local_gc_max: Optional[float] = None,

    # Existing sequence constraints
    repeat_length: Optional[int] = None,
    max_homopolymer: Optional[int] = None,
    forbidden_sites: Optional[List[str]] = None,

    # New constraints
    codon_pair_table: Optional[dict] = None,
    kmer_size: Optional[int] = None,
    max_kmer_frequency: Optional[float] = None,

    five_prime_window: Optional[int] = None,
    five_prime_gc_min: Optional[float] = None,
    five_prime_gc_max: Optional[float] = None,

    # Fitness weights
    cai_weight: float = 1.0,
    gc_weight: float = 10.0,
    target_gc_weight: float = 10.0,
    local_gc_weight: float = 5.0,
    repeat_weight: float = 5.0,
    homopolymer_weight: float = 5.0,
    restriction_weight: float = 5.0,
    edit_weight: float = 0.0,

    codon_pair_weight: float = 0.0,
    kmer_weight: float = 0.0,
    five_prime_gc_weight: float = 0.0,
) -> float:
    """
    Calculate complete GA fitness.

    Higher fitness is better.
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
    # Target GC
    # --------------------------------------------------------

    target_deviation = target_gc_deviation(
        sequence,
        target_gc,
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
    # Repeat
    # --------------------------------------------------------

    repeat_violation = repeat_penalty(
        sequence,
        repeat_length,
    )

    # --------------------------------------------------------
    # Homopolymer
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

    changes = edit_rate(
        original_codons,
        codons,
    )

    # --------------------------------------------------------
    # Codon-pair bias
    # --------------------------------------------------------

    codon_pair_violation = codon_pair_penalty(
        sequence,
        codon_pair_table,
    )

    # --------------------------------------------------------
    # K-mer complexity
    # --------------------------------------------------------

    kmer_violation = kmer_frequency_penalty(
        sequence,
        kmer_size,
        max_kmer_frequency,
    )

    # --------------------------------------------------------
    # 5' GC
    # --------------------------------------------------------

    five_prime_gc_violation = five_prime_gc_penalty(
        sequence,
        five_prime_window,
        five_prime_gc_min,
        five_prime_gc_max,
    )

    # --------------------------------------------------------
    # Fitness
    # --------------------------------------------------------

    # Global GC is treated as a strong constraint.
    HARD_CONSTRAINT_PENALTY = 1000.0

    fitness = (

        # CAI reward
        cai_weight * cai

        # Global GC
        - (
            gc_weight
            * HARD_CONSTRAINT_PENALTY
            * global_gc_violation
        )

        # Target GC
        - (
            target_gc_weight
            * target_deviation
        )

        # Local GC
        - (
            local_gc_weight
            * local_violation
        )

        # Repeats
        - (
            repeat_weight
            * repeat_violation
        )

        # Homopolymers
        - (
            homopolymer_weight
            * homopolymer_violation
        )

        # Restriction sites
        - (
            restriction_weight
            * restriction_violation
        )

        # Edit rate
        - (
            edit_weight
            * changes
        )

        # Codon-pair bias
        - (
            codon_pair_weight
            * codon_pair_violation
        )

        # K-mer complexity
        - (
            kmer_weight
            * kmer_violation
        )

        # 5' GC
        - (
            five_prime_gc_weight
            * five_prime_gc_violation
        )
    )

    return fitness


# ============================================================
# Tournament selection
# ============================================================

def tournament_selection(
    population: List[List[str]],
    fitnesses: List[float],
    rng: random.Random,
    tournament_size: int = 3,
) -> List[str]:
    """Select an individual using tournament selection."""

    if not population:
        raise ValueError(
            "Population cannot be empty."
        )

    size = min(
        max(tournament_size, 1),
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
    """Perform single-point crossover."""

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
    """Mutate codons using synonymous substitutions."""

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
# History
# ============================================================

def create_history_entry(
    generation: int,
    population: List[List[str]],
    fitnesses: List[float],
    original_codons: List[str],
    organism: str,
) -> Dict[str, Any]:
    """Create one convergence-history record."""

    best_index = max(
        range(len(population)),
        key=lambda index: fitnesses[index],
    )

    best_individual = population[
        best_index
    ]

    best_sequence = "".join(
        best_individual
    )

    return {
        "generation": generation,

        "best_fitness": max(
            fitnesses
        ),

        "mean_fitness": (
            sum(fitnesses)
            / len(fitnesses)
        ),

        "best_cai": calculate_cai(
            best_sequence,
            organism,
        ),

        "best_gc": gc_content(
            best_sequence
        ),

        "edit_rate": edit_rate(
            original_codons,
            best_individual,
        ),
    }


# ============================================================
# Main Genetic Algorithm
# ============================================================

def optimize_ga(
    dna_sequence: str,
    organism: str = "ecoli",

    # --------------------------------------------------------
    # Global GC
    # --------------------------------------------------------

    gc_min: Optional[float] = None,
    gc_max: Optional[float] = None,
    target_gc: Optional[float] = None,

    # --------------------------------------------------------
    # Local GC
    # --------------------------------------------------------

    local_gc_window: Optional[int] = None,
    local_gc_min: Optional[float] = None,
    local_gc_max: Optional[float] = None,

    # --------------------------------------------------------
    # Existing constraints
    # --------------------------------------------------------

    repeat_length: Optional[int] = None,
    max_homopolymer: Optional[int] = None,
    forbidden_sites: Optional[List[str]] = None,

    # --------------------------------------------------------
    # NEW: Codon-pair bias
    # --------------------------------------------------------

    codon_pair_table: Optional[dict] = None,

    # --------------------------------------------------------
    # NEW: K-mer
    # --------------------------------------------------------

    kmer_size: Optional[int] = None,
    max_kmer_frequency: Optional[float] = None,

    # --------------------------------------------------------
    # NEW: 5' GC
    # --------------------------------------------------------

    five_prime_window: Optional[int] = None,
    five_prime_gc_min: Optional[float] = None,
    five_prime_gc_max: Optional[float] = None,

    # --------------------------------------------------------
    # GA parameters
    # --------------------------------------------------------

    population_size: int = 100,
    generations: int = 100,

    mutation_rate: float = 0.02,
    crossover_rate: float = 0.80,

    tournament_size: int = 3,
    elitism: int = 2,

    # --------------------------------------------------------
    # Fitness weights
    # --------------------------------------------------------

    cai_weight: float = 1.0,

    gc_weight: float = 10.0,
    target_gc_weight: float = 10.0,
    local_gc_weight: float = 5.0,

    repeat_weight: float = 5.0,
    homopolymer_weight: float = 5.0,
    restriction_weight: float = 5.0,

    edit_weight: float = 0.0,

    codon_pair_weight: float = 0.0,
    kmer_weight: float = 0.0,
    five_prime_gc_weight: float = 0.0,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed: Optional[int] = None,

    # --------------------------------------------------------
    # Return convergence history
    # --------------------------------------------------------

    return_history: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    Optimize a DNA sequence using a Genetic Algorithm.

    By default returns:

        optimized_sequence

    If return_history=True:

        {
            "sequence": optimized_sequence,
            "history": [...]
        }
    """

    # ========================================================
    # Validate GA parameters
    # ========================================================

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

    # ========================================================
    # Validate global GC
    # ========================================================

    if gc_min is not None:

        if not 0.0 <= gc_min <= 1.0:
            raise ValueError(
                "gc_min must be between 0 and 1."
            )

    if gc_max is not None:

        if not 0.0 <= gc_max <= 1.0:
            raise ValueError(
                "gc_max must be between 0 and 1."
            )

    if (
        gc_min is not None
        and gc_max is not None
        and gc_min > gc_max
    ):
        raise ValueError(
            "gc_min cannot be greater than gc_max."
        )

    # ========================================================
    # Validate target GC
    # ========================================================

    if target_gc is not None:

        if not 0.0 <= target_gc <= 1.0:
            raise ValueError(
                "target_gc must be between 0 and 1."
            )

    # ========================================================
    # Validate local GC
    # ========================================================

    if local_gc_min is not None:

        if not 0.0 <= local_gc_min <= 1.0:
            raise ValueError(
                "local_gc_min must be between 0 and 1."
            )

    if local_gc_max is not None:

        if not 0.0 <= local_gc_max <= 1.0:
            raise ValueError(
                "local_gc_max must be between 0 and 1."
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

    # ========================================================
    # Validate 5' GC
    # ========================================================

    if five_prime_gc_min is not None:

        if not 0.0 <= five_prime_gc_min <= 1.0:
            raise ValueError(
                "five_prime_gc_min must be between 0 and 1."
            )

    if five_prime_gc_max is not None:

        if not 0.0 <= five_prime_gc_max <= 1.0:
            raise ValueError(
                "five_prime_gc_max must be between 0 and 1."
            )

    if (
        five_prime_gc_min is not None
        and five_prime_gc_max is not None
        and five_prime_gc_min > five_prime_gc_max
    ):
        raise ValueError(
            "five_prime_gc_min cannot be greater "
            "than five_prime_gc_max."
        )

    if (
        five_prime_window is not None
        and five_prime_window <= 0
    ):
        raise ValueError(
            "five_prime_window must be positive."
        )

    # ========================================================
    # Validate k-mer
    # ========================================================

    if kmer_size is not None:

        if kmer_size <= 0:
            raise ValueError(
                "kmer_size must be positive."
            )

    if max_kmer_frequency is not None:

        if not 0.0 <= max_kmer_frequency <= 1.0:
            raise ValueError(
                "max_kmer_frequency must be between 0 and 1."
            )

    # ========================================================
    # Prepare sequence
    # ========================================================

    dna_sequence = (
        dna_sequence
        .upper()
        .strip()
    )

    original_codons = split_codons(
        dna_sequence
    )

    original_protein = translate_dna(
        dna_sequence
    )

    # ========================================================
    # Host codon table
    # ========================================================

    codon_table = get_codon_table(
        organism
    )

    synonymous = build_synonymous_codons(
        original_codons,
        codon_table,
    )

    # ========================================================
    # Random generator
    # ========================================================

    rng = random.Random(
        seed
    )

    # ========================================================
    # Initial population
    # ========================================================

    population = initialize_population(
        original_codons,
        synonymous,
        population_size,
        rng,
    )

    # ========================================================
    # Fitness wrapper
    # ========================================================

    def fitness(
        individual: List[str],
    ) -> float:

        return calculate_fitness(

            individual,

            original_codons,

            organism,

            # Global GC
            gc_min,
            gc_max,

            # Target GC
            target_gc,

            # Local GC
            local_gc_window,
            local_gc_min,
            local_gc_max,

            # Existing constraints
            repeat_length,
            max_homopolymer,
            forbidden_sites,

            # Codon pair
            codon_pair_table,

            # K-mer
            kmer_size,
            max_kmer_frequency,

            # 5' GC
            five_prime_window,
            five_prime_gc_min,
            five_prime_gc_max,

            # Weights
            cai_weight,
            gc_weight,
            target_gc_weight,
            local_gc_weight,
            repeat_weight,
            homopolymer_weight,
            restriction_weight,
            edit_weight,

            codon_pair_weight,
            kmer_weight,
            five_prime_gc_weight,
        )

    # ========================================================
    # Initial fitness
    # ========================================================

    fitnesses = [
        fitness(individual)
        for individual in population
    ]

    # ========================================================
    # History
    # ========================================================

    history: List[Dict[str, Any]] = []

    history.append(
        create_history_entry(
            generation=0,
            population=population,
            fitnesses=fitnesses,
            original_codons=original_codons,
            organism=organism,
        )
    )

    # ========================================================
    # Initial best
    # ========================================================

    best_index = max(
        range(len(population)),
        key=lambda index: fitnesses[index],
    )

    best_individual = population[
        best_index
    ].copy()

    best_fitness = fitnesses[
        best_index
    ]

    # ========================================================
    # Evolution
    # ========================================================

    for generation in range(
        1,
        generations + 1,
    ):

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
            for index in ranked_indices[
                :elite_count
            ]
        ]

        # ----------------------------------------------------
        # Create children
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

            # Crossover
            if rng.random() < crossover_rate:

                child1, child2 = crossover(
                    parent1,
                    parent2,
                    rng,
                )

            else:

                child1 = parent1.copy()
                child2 = parent2.copy()

            # Mutation
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

            new_population.append(
                child1
            )

            if len(new_population) < population_size:

                new_population.append(
                    child2
                )

        # ----------------------------------------------------
        # Replace population
        # ----------------------------------------------------

        population = new_population

        # ----------------------------------------------------
        # Evaluate population
        # ----------------------------------------------------

        fitnesses = [
            fitness(individual)
            for individual in population
        ]

        # ----------------------------------------------------
        # Current best
        # ----------------------------------------------------

        current_best_index = max(
            range(len(population)),
            key=lambda index: fitnesses[index],
        )

        current_best_fitness = fitnesses[
            current_best_index
        ]

        # ----------------------------------------------------
        # Update global best
        # ----------------------------------------------------

        if current_best_fitness > best_fitness:

            best_fitness = (
                current_best_fitness
            )

            best_individual = population[
                current_best_index
            ].copy()

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history.append(
            create_history_entry(
                generation=generation,
                population=population,
                fitnesses=fitnesses,
                original_codons=original_codons,
                organism=organism,
            )
        )

    # ========================================================
    # Candidate sequence
    # ========================================================

    optimized_sequence = "".join(
        best_individual
    )

    # ========================================================
    # Global GC feasibility
    # ========================================================

    if not is_gc_feasible(
        optimized_sequence,
        gc_min,
        gc_max,
    ):

        feasible_candidates = []

        for individual in population:

            candidate = "".join(
                individual
            )

            if is_gc_feasible(
                candidate,
                gc_min,
                gc_max,
            ):

                feasible_candidates.append(
                    individual
                )

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
                "the requested global GC constraint."
            )

    # ========================================================
    # Protein verification
    # ========================================================

    verification_result = verify_translation(
        dna_sequence,
        optimized_sequence,
    )

    if not verification_result["valid"]:

        raise RuntimeError(
            "GA produced a sequence that does not "
            "preserve the original protein."
        )

    # ========================================================
    # Final protein check
    # ========================================================

    final_protein = translate_dna(
        optimized_sequence
    )

    if final_protein != original_protein:

        raise RuntimeError(
            "Protein sequence was not preserved."
        )

    # ========================================================
    # Return
    # ========================================================

    if return_history:

        return {
            "sequence": optimized_sequence,
            "history": history,
        }

    return optimized_sequence


# ============================================================
# Compatibility wrapper
# ============================================================

def optimize_sequence_ga(
    dna_sequence: str,
    organism: str = "ecoli",
    **kwargs,
) -> Union[str, Dict[str, Any]]:
    """
    Compatibility wrapper around optimize_ga().
    """

    return optimize_ga(
        dna_sequence,
        organism=organism,
        **kwargs,
    )