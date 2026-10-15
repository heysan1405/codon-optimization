"""
mRNA secondary-structure utilities using ViennaRNA.
"""

from typing import Dict, Optional


def dna_to_rna(sequence: str) -> str:
    """Convert DNA sequence to RNA."""
    return sequence.upper().replace("T", "U")


def fold_mrna(sequence: str) -> Dict[str, object]:
    """
    Fold an mRNA sequence using ViennaRNA.

    Returns:
        structure: Dot-bracket secondary structure
        mfe: Minimum Free Energy in kcal/mol
        length: RNA length
    """

    try:
        import RNA
    except ImportError as exc:
        raise ImportError(
            "ViennaRNA is required. Install with: "
            "python -m pip install viennarna"
        ) from exc

    rna = dna_to_rna(sequence)

    if not rna:
        return {
            "structure": "",
            "mfe": 0.0,
            "length": 0,
        }

    structure, mfe = RNA.fold(rna)

    return {
        "structure": structure,
        "mfe": float(mfe),
        "length": len(rna),
    }


def mfe(sequence: str) -> float:
    """Return the minimum free energy of an mRNA sequence."""
    return float(fold_mrna(sequence)["mfe"])


def structure(sequence: str) -> str:
    """Return ViennaRNA dot-bracket structure."""
    return str(fold_mrna(sequence)["structure"])


def mfe_penalty(
    sequence: str,
    target_mfe: Optional[float] = None,
    min_mfe: Optional[float] = None,
    max_mfe: Optional[float] = None,
) -> float:
    """
    Calculate a secondary-structure penalty.

    If target_mfe is supplied:
        penalty = absolute difference from target MFE.

    Otherwise:
        penalty is the amount outside [min_mfe, max_mfe].
    """

    value = mfe(sequence)

    if target_mfe is not None:
        return abs(value - target_mfe)

    penalty = 0.0

    if min_mfe is not None and value < min_mfe:
        penalty += min_mfe - value

    if max_mfe is not None and value > max_mfe:
        penalty += value - max_mfe

    return penalty
