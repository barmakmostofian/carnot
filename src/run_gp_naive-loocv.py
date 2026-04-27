"""
This script runs GP naively, i.e., removing one compound at a time, 
refitting the GP on the remaining n-1 compounds, and predicting the 
held-out compound.

The corresponding GP equations in matrix notation are:
μ* = k*ᵀ · (K + σ²ₙI)⁻¹ · y
σ²* = k** − kᵀ · (K + σ²ₙI)⁻¹ · k*
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
# Construct and check Kriging matrix, K 
# ------------------------------------------------------------------

K = SIGMA2_F * T + SIGMA2_N * np.eye(n)

print(f"\nConstructed Kriging matrix, K = {SIGMA2_F} * T + {SIGMA2_N} * I")
print(f"  K diagonal (should be {SIGMA2_F + SIGMA2_N:.4f} everywhere):")
print(f"  {np.round(np.diag(K), 4)}")

# Check that K is positive definite.
check_pd(K)



# ------------------------------------------------------------------
# Run naive LOO loop
# ------------------------------------------------------------------

# For each held-out compound i, the remaining compounds form the Kriging matrix, 
# K_train, and the observed potencies, y_train. The Tanimoto similarities between 
# the training set and the test compound is the vector k_star. The noisy self-
# similarity of the test compound is k_starstar. The variable names follow the GP 
# equation notation in the header.


print(f"\nRunning naive LOO cross-validation ({n} folds) ...")

loo_mu    = np.zeros(n)   # posterior mean values for each held-out compound
loo_sigma = np.zeros(n)   # posterior std dev values for each held-out compound


for i in range(n):

    # Return the n-1 indices that remain when compound i is left out.
    train_idx = np.array([j for j in range(n) if j != i])

    K_train    = K[np.ix_(train_idx, train_idx)]    # (n-1) x (n-1) matrix
    k_star     = K[i, train_idx]                    # (n-1) x 1 vector
    k_starstar = K[i, i]                            # scalar
    y_train    = y[train_idx]                       # (n-1) x 1 vector

    # Cholesky factorisation of this fold's training matrix
    K_factor_train, Lower_tri_train = factorize(K_train, lower=True)

    # alpha_train = K_train^{-1} . y_train
    alpha_train = linalg.cho_solve((K_factor_train, Lower_tri_train), y_train)

    # Posterior mean (see header) 
    # It is centered around 0, we add back mean_y
    mu_centred  = float(k_star @ alpha_train)
    loo_mu[i]   = mu_centred + mean_y

    # Posterior variance (see header)
    Kinv_kstar    = linalg.cho_solve((K_factor_train, Lower_tri_train), k_star)
    var         = k_starstar - float(k_star @ Kinv_kstar)
    var         = max(var, 0.0)   # numerical safety: clip to 0
    loo_sigma[i]  = np.sqrt(var)

print("  Done.")



