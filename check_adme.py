import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from utils import desirability
from utils import cns_mpo



############################################################
######### Simple CNS MPO (Wager et al. 2010, 2016) #########

file = 'example_compounds.csv'
df = pd.read_csv(file, sep=',', header=0)


# Long version:
for mol in df['Compound Structure'] :
    result = cns_mpo(mol)
    for key, value in result.items():
        print(f"{key}: {value}")
    print("\n")


# Short version:
print("compound \t\t mpo score")
for mol in df['Compound Structure'] :
    result = cns_mpo(mol)
    print(mol, result['MPO_score'])

############################################################
