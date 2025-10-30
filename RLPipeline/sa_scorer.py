"""
Synthetic Accessibility (SA) score utility using the official RDKit implementation.

This module delegates SA score computation to RDKit's contrib module
`rdkit.Chem.SA_Score.sascorer.calculateScore`, which implements the
Ertl & Schuffenhauer (2009) method. The score is on a ~1–10 scale where
lower indicates easier synthesis.

API:
    - calculate_sa_score(mol: Chem.Mol) -> float
    - safe_sa_from_smiles(smiles: str) -> float | None

Reference:
    Ertl, P.; Schuffenhauer, A. "Estimation of synthetic accessibility score
    of drug-like molecules based on molecular complexity and fragment
    contributions." J. Cheminf 2009, 1, 8.
"""
from __future__ import annotations
from typing import Optional
from rdkit import Chem

# Prefer the official RDKit SA_Score implementation
try:
    from rdkit.Contrib import SA_Score as _SA_Score
    _HAS_OFFICIAL_SA = True
except Exception:  # pragma: no cover - environment without SA_Score module
    _HAS_OFFICIAL_SA = False


def calculate_sa_score(mol: Chem.Mol) -> float:
    """Return the RDKit SA_Score for a molecule using the official scorer.

    Raises:
        ImportError: if RDKit's SA_Score module is not available.
        ValueError: if mol is None or sanitization fails.
    """
    if mol is None:
        raise ValueError("calculate_sa_score: mol is None")
    # Ensure the molecule is sanitizable (official scorer expects valid mol)
    Chem.SanitizeMol(mol)
    if not _HAS_OFFICIAL_SA:
        raise ImportError(
            "RDKit SA_Score module not available. Please install RDKit with contrib modules "
            "so that `from rdkit.Chem import SA_Score` works."
        )
    # Official RDKit implementation
    return float(_SA_Score.sascorer.calculateScore(mol))


def safe_sa_from_smiles(smiles: str) -> Optional[float]:
    """Return SA score for a SMILES using RDKit's official scorer, or None if invalid/unavailable."""
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        return calculate_sa_score(m)
    except Exception:
        return None
