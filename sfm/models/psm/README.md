## PSM: Physics Science Module

This module is responsible for the unified structure encoder of protein, small molecule, and periodic crystal.

### Data
All processed data is stored in: https://hai1data.blob.core.windows.net/sfm/psm/

### scripts
To run models on your local machine, try bash scripts in `./scripts/psm` folder, i.e.:
```bash
bash scripts/psm/pretrain_psm.sh
```

### amlt yaml
To run jobs on cluster, use yaml file in the `./amlt/psm` folder, i.e.:
```bash
amlt project checkout psm
amlt run ./amlt/psm/PSM150M.yaml PSMv0_150M
```
If you do not have a psm project, you can create one by:
```bash
amlt project create psm hai1data
```
if you do not have amlt installed, you can install it by:
```bash
pip install -U "amlt>=10.0.3.dev0" --index-url https://msrpypi.azurewebsites.net/nightly/leloojoo
```

### Data ratio
To use differnt data ratio, you can change the `data_ratio` in the yaml file or scripts, e.g.:
```yaml
dataset_split_raito='0.4,0.2,0.4'
```
means 40% for small molecule, 20% for periodic crystal, and 40% for protein.
