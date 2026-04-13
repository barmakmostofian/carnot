from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np

def desirability(val, low_ideal, high_ideal, low_zero=None, high_zero=None):
    """Piecewise linear desirability function → score between 0 and 1."""
    if low_zero is not None and val <= low_zero:
        return 0.0
    if high_zero is not None and val >= high_zero:
        return 0.0
    if low_ideal <= val <= high_ideal:
        return 1.0
    if val < low_ideal and low_zero is not None:
        return (val - low_zero) / (low_ideal - low_zero)
    if val > high_ideal and high_zero is not None:
        return (high_zero - val) / (high_zero - high_ideal)
    return 1.0

def cns_mpo(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # --- Compute the 6 properties ---
    mw    = Descriptors.MolWt(mol)
    clogp = Descriptors.MolLogP(mol)
    clogd = clogp - 0.75  # Simplified
    tpsa  = rdMolDescriptors.CalcTPSA(mol)
    hbd   = rdMolDescriptors.CalcNumHBD(mol)
    pka   = 8.0  # Placeholder; requires external pKa calculator (e.g. oechem)

    # --- Desirability functions (Wager et al. 2010) ---
    scores = {
        "cLogP": desirability(clogp, low_ideal=-np.inf, high_ideal=3,  low_zero=None, high_zero=5),
        "cLogD": desirability(clogd, low_ideal=-np.inf, high_ideal=2,  low_zero=None, high_zero=4),
        "MW":    desirability(mw,    low_ideal=0,        high_ideal=360, low_zero=None, high_zero=500),
        "TPSA":  desirability(tpsa,  low_ideal=40,       high_ideal=90,  low_zero=20,   high_zero=120),
        "HBD":   desirability(hbd,   low_ideal=0,        high_ideal=0.5, low_zero=None, high_zero=3.5),
        "pKa":   desirability(pka,   low_ideal=-np.inf,  high_ideal=8,   low_zero=None, high_zero=10),
    }

    mpo_score = sum(scores.values())

    return {
        "SMILES": smiles,
        "properties": {"MW": mw, "cLogP": clogp, "cLogD": clogd,
                       "TPSA": tpsa, "HBD": hbd, "pKa": pka},
        "component_scores": scores,
        "MPO_score": round(mpo_score, 3),
        "CNS_favorable": mpo_score >= 4.0
    }

# --- Example ---
result = cns_mpo("CN1CCC[C@H]1c2cccnc2")  # Nicotine
for k, v in result.items():
    print(f"{k}: {v}")
