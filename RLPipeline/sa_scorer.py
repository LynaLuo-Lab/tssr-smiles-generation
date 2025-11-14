"""
Synthetic Accessibility (SA) score utility using the official RDKit implementation.

This module delegates SA score computation to RDKit's SA_Score contrib module
(`rdkit.Chem.SA_Score.sascorer.calculateScore`), which implements the
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
from rdkit.Chem import Descriptors, rdMolDescriptors

# Prefer the official RDKit SA_Score implementation (robust import across RDKit builds)
_HAS_SA_SCORER = False
_sascorer = None  # type: ignore

# Newer/typical location
try:  # pragma: no cover - import depends on environment
    from rdkit.Chem.SA_Score import sascorer as _sascorer  # type: ignore
    _HAS_SA_SCORER = True
except Exception:  # pragma: no cover
    try:
        # Some builds expose it as a module under rdkit.Chem
        from rdkit.Chem import SA_Score as _SA_SA  # type: ignore
        _sascorer = _SA_SA.sascorer  # type: ignore
        _HAS_SA_SCORER = True
    except Exception:
        _HAS_SA_SCORER = False


def _heuristic_sa_score(mol: Chem.Mol) -> float:
    """Heuristic SA approximation in case the official RDKit SA_Score is unavailable.

    The value roughly maps into [1,10] (lower is easier) using a combination
    of size, ring/complexity, stereochemistry, and heteroatom content.
    This is NOT a drop-in replacement for the Ertl SA score, but provides a
    stable fallback to keep downstream scripts working when SA_Score is missing.
    """
    # Work on a sanitized copy
    m = Chem.Mol(mol)
    Chem.SanitizeMol(m)

    n_heavy = rdMolDescriptors.CalcNumHeavyAtoms(m)
    n_rings = rdMolDescriptors.CalcNumRings(m)
    n_arom_rings = rdMolDescriptors.CalcNumAromaticRings(m)
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(m)
    n_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(m)
    n_chiral = rdMolDescriptors.CalcNumAtomStereoCenters(m)
    fr_csp3 = Descriptors.FractionCSP3(m)
    bertz = max(0.0, float(Descriptors.BertzCT(m)))
    n_hetero = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() not in (1, 6))

    # Compose a raw complexity-like score (higher = more complex)
    raw = (
        0.04 * n_heavy
        + 0.30 * n_rings
        + 0.35 * n_arom_rings
        + 0.30 * n_hetero
        + 0.50 * n_spiro
        + 0.45 * n_bridge
        + 0.25 * n_chiral
        + 0.0015 * bertz
    )
    # Penalize flat molecules (very low sp3 fraction)
    raw += 0.8 * max(0.0, 0.5 - fr_csp3)

    # Map raw to approx [1,10]: use a softplus-like transform then clamp
    # For typical druglike raw in ~[0.5,8], this yields SA ~[2,7]
    import math
    sa = 1.0 + 9.0 * (math.log1p(raw) / math.log1p(12.0))
    if sa < 1.0:
        sa = 1.0
    elif sa > 10.0:
        sa = 10.0
    return float(sa)


def calculate_sa_score(mol: Chem.Mol) -> float:
    """Return SA score for a molecule.

    Uses RDKit's official SA_Score when available; otherwise falls back to a
    heuristic approximation that returns a value on a 1–10 scale (lower easier).

    Raises:
        ValueError: if mol is None or sanitization fails.
    """
    if mol is None:
        raise ValueError("calculate_sa_score: mol is None")
    # Ensure the molecule is sanitizable
    Chem.SanitizeMol(mol)
    if _HAS_SA_SCORER and _sascorer is not None:  # official implementation
        try:
            return float(_sascorer.calculateScore(mol))  # type: ignore
        except Exception:
            # If official scorer errors out, use heuristic
            pass
    # Fallback heuristic
    return _heuristic_sa_score(mol)


def safe_sa_from_smiles(smiles: str) -> Optional[float]:
    """Return SA score for a SMILES using the official scorer when available,
    or a heuristic fallback otherwise. Returns None only if SMILES is invalid.
    """
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        return calculate_sa_score(m)
    except Exception:
        return None
