import numpy as np

from rdkit.Chem import MolFromSmiles, Descriptors, rdMolDescriptors, RDKFingerprint


###############################################################################################
# ADME properties based on CNS MPO desirability scores (Wager et al., 2010, 2016) 
###############################################################################################

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
    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Derive the 6 properties
    mw    = Descriptors.MolWt(mol)
    clogp = Descriptors.MolLogP(mol)
    clogd = clogp - 0.75  # Simplified
    tpsa  = rdMolDescriptors.CalcTPSA(mol)
    hbd   = rdMolDescriptors.CalcNumHBD(mol)
    pka   = 8.0  # Placeholder; requires external pKa calculator (e.g. epik, molgpka, or qupkake)

    # Compute desirability scores
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

###############################################################################################




###############################################################################################
# Check the validity of a Tanimoto similarity matrix to serve as a Kernel in GPs 
###############################################################################################

# Check that the matrix is positive semi-definite indentifying its smallest eigenvalue
def check_psd(matrix) :
    eigenvalues = np.linalg.eigvalsh(matrix)   
    print(f"\nEigenvalue check (Matrix should be positive semi-definite):")
    print(f"  Smallest eigenvalue : {eigenvalues.min():.6f}")
    print(f"  Largest eigenvalue  : {eigenvalues.max():.6f}")
    print(f"  Number of eigenvalues < 1e-6: "f"{np.sum(eigenvalues < 1e-6)}")

    if eigenvalues.min() < -1e-6:
        print("\n  WARNING: Matrix has negative eigenvalues — not PSD.")
    else:
        print("\n  Matrix is positive semi-definite.\n  PSD check passed!")



# Check that this is a symmetric matrix with simple assertions
def check_unit_symmetry(matrix) :
    assert np.allclose(matrix, matrix.T), "Matrix is not symmetric!"
    assert np.allclose(np.diag(matrix), 1.0), "Diagonal is not 1.0!"
    print("\nSymmetry check  : passed!")
    print("Diagonal check  : passed!")



# Check for pairs with Tanimoto = 1.0 (which are structurally identical based on their fingerprints)
def get_identical_pairs(matrix) :

    n = matrix.shape[0]

    identical_pairs = [(i+1, j+1) for i in range(n) for j in range(i+1, n)
                   if matrix[i, j] >= 0.9999]
    if identical_pairs:
        print(f"\n  Pairs with Tanimoto = 1.000 (identical fingerprints):")
        for a, b in identical_pairs:
            print(f"    mol_{a:02d} and mol_{b:02d}")
    else :
        print("  There are no pairs of identical structures based on their fingerprints!")



# Print all values, then the diagonal, and some off-diagonal metrics
def echo_matrix(matrix) :
    print("\nTanimoto similarity matrix:")

    n = matrix.shape[0]

    col_names = "\t\t" + "  ".join(f"mol_{j+1:02d}" for j in range(n))
    print(col_names)

    for i in range(n):
        row_name = f"mol_{i+1:02d}"
        row_vals = "  ".join(f"{matrix[i,j]:.3f}" for j in range(n))
        print(f"{row_name}\t\t{row_vals}")


    print(f"\nDiagonal entries (should all be 1.000):")
    print("  " + "  ".join(f"{v:.3f}" for v in np.diag(matrix)))


    upper = matrix[np.triu_indices(n, k=1)]
    print(f"\nOff-diagonal summary:")
    print(f"  Min  : {upper.min():.4f}")
    print(f"  Max  : {upper.max():.4f}")
    print(f"  Mean : {upper.mean():.4f}")

############################################################################################################
