import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from rdkit import DataStructs


# ------------------------------------------------------------------
# Define constants — edit these to match your file and preferences
# ------------------------------------------------------------------

CSV_FILE    = "example_compounds.csv"   # path to your input CSV file
SMI_COL     = "compound structure"      # column name for smiles strings
PIC50_COL   = "pic50"                   # column name for pIC50 values
FP_SIZE     = 1024                      # number of bits in the fingerprint
MIN_PATH    = 1                         # minimum path length (bonds)
MAX_PATH    = 7                         # maximum path length (bonds)



# ------------------------------------------------------------------
# Read data
# ------------------------------------------------------------------

print(f"Reading data from '{CSV_FILE}' ...")
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"  File '{CSV_FILE}' not found.")

print(f"  Loaded {len(df)} compounds.")
print(f"  Columns found: {list(df.columns)}")


# Convert data to arrays for downstream use
smiles_list = df[SMI_COL].tolist()
pic50       = df[PIC50_COL].to_numpy(dtype=float)
n           = len(smiles_list)



# ------------------------------------------------------------------
# Compute path-based fingerprints (RDKit)
# ------------------------------------------------------------------
# Chem.MolFromSmiles() parses and sanitises the SMILES.
# RDKFingerprint() computes a Daylight-style path-based fingerprint:
#   - enumerates all linear paths from minPath to maxPath bonds
#   - hashes each path into a bit position

print(f"\nComputing RDKit path-based fingerprints "
      f"(fpSize={FP_SIZE}, maxPath={MAX_PATH}) ...")

mols = []
fps  = []
failed = []

for idx, smi in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"  WARNING: could not parse SMILES at row {idx}: {smi}")
        failed.append(idx)
        continue
    fp = RDKFingerprint(mol, minPath=MIN_PATH, maxPath=MAX_PATH, fpSize=FP_SIZE)
    mols.append(mol)
    fps.append(fp)

if failed:
    # Drop rows with unparseable SMILES
    df      = df.drop(index=failed).reset_index(drop=True)
    pic50   = np.delete(pic50, failed)
    n       = len(fps)
    print(f"  {len(failed)} compound(s) dropped due to invalid SMILES.")

print(f"  Done. {n} fingerprints computed.")
print(f"  Bits set in fingerprint 1: {fps[0].GetNumOnBits()} / {FP_SIZE}")



# ------------------------------------------------------------------
# Build the Tanimoto similarity matrix, T,and prove that it is a kernel
# ------------------------------------------------------------------
# DataStructs.BulkTanimotoSimilarity(fp_i, fp_list) computes the
# Tanimoto similarity between fp_i and every fingerprint in fp_list
# in a single efficient call. We use this to fill row i of T.

print("\nBuilding Tanimoto kernel matrix T ...")

T = np.zeros((n, n))
for i in range(n):
    row = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
    T[i, :] = row

print("...done.")



# Print all values, then the diagonal, and some off-diagonal metrics
print("\nTanimoto matrix:")
col_names = "\t\t" + "  ".join(f"mol_{j+1:02d}" for j in range(n))
print(col_names)
for i in range(n):
    row_name = f"mol_{i+1:02d}"
    row_vals = "  ".join(f"{T[i,j]:.3f}" for j in range(n))
    print(f"{row_name}\t\t{row_vals}")


print(f"\nDiagonal entries (should all be 1.000):")
print("  " + "  ".join(f"{v:.3f}" for v in np.diag(T)))


upper = T[np.triu_indices(n, k=1)]
print(f"\nOff-diagonal summary:")
print(f"  Min  : {upper.min():.4f}")
print(f"  Max  : {upper.max():.4f}")
print(f"  Mean : {upper.mean():.4f}")



# Check for pairs with Tanimoto = 1.0 (which are structurally identical based on their fingerprints)
identical_pairs = [(i+1, j+1) for i in range(n) for j in range(i+1, n)
                   if T[i, j] >= 0.9999]
if identical_pairs:
    print(f"\n  Pairs with T = 1.000 (identical fingerprints):")
    for a, b in identical_pairs:
        print(f"    mol_{a:02d} and mol_{b:02d}")
    print("  Note: identical rows make the matrix singular — this confirms")
    print("  that adding noise is essential for invertibility.")


# Check for symmetry and diagonal assertions
assert np.allclose(T, T.T),         "T is not symmetric."
assert np.allclose(np.diag(T), 1.0),"Diagonal is not 1.0."
print("\nSymmetry check  : passed")
print("Diagonal check  : passed")


# Check that T is positive-definite
eigenvalues = np.linalg.eigvalsh(T)   # eigvalsh is for symmetric matrices
print(f"\nEigenvalue check (T should be positive semi-definite):")
print(f"  Smallest eigenvalue : {eigenvalues.min():.6f}")
print(f"  Largest eigenvalue  : {eigenvalues.max():.6f}")
print(f"  Number of eigenvalues < 1e-6: "
      f"{np.sum(eigenvalues < 1e-6)}")

if eigenvalues.min() < -1e-6:
    print("  WARNING: T has negative eigenvalues — not PSD.")
else:
    print("  T is positive semi-definite. PSD check passed.")


# Save Tanimoto kernel
np.save("tanimoto_matrix.npy", T)


