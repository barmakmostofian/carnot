"""
This script runs GP naively, i.e., removing one compound at a time, 
refitting the GP on the remaining n-1 compounds, and predicting the 
held-out compound.
"""


import sys
import numpy as np
import pandas as pd
from scipy import linalg

from utils import *



# ------------------------------------------------------------------
# Define constants — edit these to match your file and preferences
# ------------------------------------------------------------------

CSV_FILE    = "example_compounds.csv"   # path to your input CSV file
SMI_COL     = "Compound Structure"      # column name for smiles strings
OBS_COL     = "pic50"                   # column name for observed values
SIGMA2_F    = 1.0                       # signal variance  (scales the kernel amplitude)
SIGMA2_N    = 0.1                       # noise variance   (added to diagonal only)



# ------------------------------------------------------------------
# Read and preprocess data
# ------------------------------------------------------------------

print(f"\nLoading data ...")
try:
    df = pd.read_csv(CSV_FILE)
    T     = np.load("tanimoto_matrix.npy")
except FileNotFoundError:
    raise SystemExit(
        "Could not find tanimoto_matrix.npy or '{CSV_FILE}'.\n"
    )

obs_list    = df[OBS_COL].to_numpy(dtype=float)
n           = len(obs_list)
if T.shape[0] != n :
    sys.exit("Error: tanimoto matrix seems to have the wrong shape!")
else :
    print(f"  Loaded tanimoto matrix, T ({n} x {n}), and vector of ({n}) observed target values.")


# Removing the mean of the observed values, against which we train the model, since the (uninformed) GP prior has zero mean.
# After making predictions in centered space, the mean can be added back when reporting results.
mean_y = obs_list.mean()
y      = obs_list - mean_y

print(f"\nObserved values, y:")
print(f"  Arithmetic mean: {mean_y:.4f}")
print(f"  Centered values: {np.round(y, 3)}")



# ------------------------------------------------------------------
# Construct and check Kriging matrix
# ------------------------------------------------------------------

C = SIGMA2_F * T + SIGMA2_N * np.eye(n)

print(f"\nConstructed Kriging matrix, C = {SIGMA2_F} * T + {SIGMA2_N} * I")
print(f"  C diagonal (should be {SIGMA2_F + SIGMA2_N:.4f} everywhere):")
print(f"  {np.round(np.diag(C), 4)}")

# Check that C is positive definite.
check_pd(C)



# ------------------------------------------------------------------
# Run naive LOO loop
# ------------------------------------------------------------------



