## Processed PubChemQC-B3LYP-PM6 dataset
### 20240527.1
Remove molecules with invalid energy label (energy $\geq$ 0). `77034910/86059366 (89.51%)` molecules were extracted.
```json
{
  "version": "20240527.1",
  "config": {
    "remove_smiles_mismatch": true,
    "ignore_isomeric_mismatch": true,
    "remove_multi_frags": true,
    "compress_graph": true
  },
  "counts": {
    "SanitizeError-AtomValenceException": 2682561,
    "SmilesMatchError-Unknown": 3871268,
    "EnergyError": 32527,
    "UnknownError": 17149,
    "MultiFragsError": 2420951,
    "valid": 77034910
  }
}
```

### 20240417.1
`77067437/86059366 (89.55%)` molecules were extracted.
```json
{
  "version": "20240417.1",
  "config": {
    "remove_smiles_mismatch": true,
    "ignore_isomeric_mismatch": true,
    "remove_multi_frags": true,
    "compress_graph": true
  },
  "counts": {
    "SanitizeError-AtomValenceException": 2682561,
    "SmilesMatchError-Unknown": 3871268,
    "UnknownError": 17149,
    "MultiFragsError": 2420951,
    "valid": 77067437
  }
}
```

## How to build PubChemQC-B3LYP-PM6 dataset
```bash
conda activate sfm
export DATA_VERSION=20240527.1
PYTHONPATH=. python sfm/data/mol_data/pubchemqc/b3lyp_pm6.py --data-dir /data/psm/PubChemQC-B3LYP-PM6/raw/Compounds --output-dir /data/psm/PubChemQC-B3LYP-PM6 --version ${DATA_VERSION} --workdir /mnt/pm6
azcopy copy "/data/psm/PubChemQC-B3LYP-PM6/${DATA_VERSION}/full/*" "https://hai1data.blob.core.windows.net/sfm/psm/PubChemQC-B3LYP-PM6/${DATA_VERSION}/full?<sas-token>" --recursive
```
