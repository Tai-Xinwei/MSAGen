export SFM_PATH=$HOME/SFM_framework
export EXAMPLE_PATH=$HOME/Fairseq/cube_examples/ai4sci

export NPROC_PER_NODE=$(nvidia-smi -L | wc -l)

if [[ -z "${OMPI_COMM_WORLD_SIZE}" ]]; then
    export NNODES=1
else
    export NNODES=${OMPI_COMM_WORLD_SIZE}
fi

if [[ -z "${OMPI_COMM_WORLD_SIZE}" ]]; then
    export NODE_RANK=0
else
    export NODE_RANK=${OMPI_COMM_WORLD_RANK}
fi

export GRAPHORMER_CKPT_NAME="pretrain56w_pm6oc_512_pm6_oc2_acc2_4UnifiedH_22B_L46_4e4_global_step560000_merged.pt"
export GRAPHORMER_CKPT_PATH=$HOME/$GRAPHORMER_CKPT_NAME
export LLAMA_CKPT_PATH=$HOME/llama2/llama-2-70b-hf

export DATA_PATH=$HOME/chemical-copilot-special-token-20231129

# seqlen
export TOKENS_PER_SAMPLE=1024
# batch size for each scale unit
export BATCH_SIZE=1
# gradient accumulation times
export UPDATE_FREQ=16
# warmup steps
export WARMUP_UPDATES=12000
# max training steps
export MAX_UPDATES=50000
# seed for model training
export SEED=12345

# 256 gpu / 64
export ZERO_N_GROUPS=4

############################################################################

if [ "$STAGE" -eq 1 ]; then
    export MAX_NUM_MOL_PER_SAMPLE=1
    export DATASET_NAMES="mol-instruction-mol-desc,chebi,functional-group,func-group-list-and-desc"
    export DATASET_SPLITS="clean,all,all,all"
    export DATASET_RATIOS="3.0,3.0,1.0,1.0"
    export FULL_MODEL_CKPT_PATH=""
elif [ "$STAGE" -eq 2 ]; then
    export MAX_NUM_MOL_PER_SAMPLE=2
    export DATASET_NAMES="mol-instruction-mol-desc,chebi,functional-group,func-group-list-and-desc,chemcop-instruction,tdc/LD50_Zhu,tdc/kcnq2_potassium_channel_butkiewicz,tdc/Skin_Reaction,tdc/HIV,tdc/CYP3A4_Veith,tdc/CYP1A2_Veith,tdc/hERG_Karim,tdc/PAMPA_NCATS,tdc/hERG,tdc/CYP2C9_Substrate_CarbonMangels,tdc/HydrationFreeEnergy_FreeSolv,tdc/m1_muscarinic_receptor_agonists_butkiewicz,tdc/Bioavailability_Ma,tdc/m1_muscarinic_receptor_antagonists_butkiewicz,tdc/DILI,tdc/potassium_ion_channel_kir2.1_butkiewicz,tdc/CYP2C9_Veith,tdc/SARSCoV2_3CLPro_Diamond,tdc/Clearance_Hepatocyte_AZ,tdc/choline_transporter_butkiewicz,tdc/Half_Life_Obach,tdc/Lipophilicity_AstraZeneca,tdc/cav3_t-type_calcium_channels_butkiewicz,tdc/SARSCoV2_Vitro_Touret,tdc/Caco2_Wang,tdc/VDss_Lombardo,tdc/PPBR_AZ,tdc/Solubility_AqSolDB,tdc/tyrosyl-dna_phosphodiesterase_butkiewicz,tdc/Carcinogens_Lagunin,tdc/Pgp_Broccatelli,tdc/CYP2C19_Veith,tdc/CYP3A4_Substrate_CarbonMangels,tdc/CYP2D6_Substrate_CarbonMangels,tdc/serine_threonine_kinase_33_butkiewicz,tdc/orexin1_receptor_butkiewicz,tdc/AMES,tdc/CYP2D6_Veith,tdc/Tox21/NR-AR,tdc/Tox21/NR-PPAR-gamma,tdc/Tox21/NR-AR-LBD,tdc/Tox21/NR-Aromatase,tdc/Tox21/SR-MMP,tdc/Tox21/NR-AhR,tdc/Tox21/SR-HSE,tdc/Tox21/NR-ER,tdc/Tox21/SR-ARE,tdc/Tox21/NR-ER-LBD,tdc/Tox21/SR-p53,tdc/Tox21/SR-ATAD5,tdc/herg_central/hERG_inhib,tdc/herg_central/hERG_at_1uM,tdc/herg_central/hERG_at_10uM,tdc/USPTO_Yields,tdc/Buchwald-Hartwig"
    export DATASET_SPLITS="clean,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all,all"
    export DATASET_RATIOS="5.0,5.0,5.0,5.0,1.0,16.93,0.41,309.60,3.04,10.14,9.52,9.30,61.46,191.20,187.27,194.55,2.02,195.31,2.02,263.85,0.41,10.34,142.05,103.09,0.41,187.97,29.76,1.24,84.25,137.36,110.62,44.82,12.52,0.37,446.43,102.77,10.34,186.92,187.97,0.39,0.57,17.18,9.52,17.21,19.38,18.50,21.47,21.51,19.09,19.33,20.19,21.43,17.97,18.45,17.67,0.41,0.41,0.41,0.15,31.61"
    export FULL_MODEL_CKPT_PATH="$HOME/ai4sci_22b_70b_stage1/checkpoint/pytorch_model.bin"
elif [ "$STAGE" -eq 3 ]; then
    export MAX_NUM_MOL_PER_SAMPLE=3
    export DATASET_NAMES="pubmed-instruction/smiles,pubmed-instruction/no-smiles,stackexchange/bioinformatics,stackexchange/biology,stackexchange/chemistry,stackexchange/cogsci,stackexchange/health,stackexchange/materials,stackexchange/physics"
    export DATASET_SPLITS="all,all,all,all,all,all,all,all,all"
    export DATASET_RATIOS="1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0"
    export FULL_MODEL_CKPT_PATH="$HOME/ai4sci_22b_70b_stage2/checkpoint/pytorch_model.bin"
else
    echo "Must set environment value STAGE in {1, 2, 3}!"
    exit 1
fi

############################################################################

STAGE_FOLDER=ai4sci_22b_70b_stage$STAGE
mkdir $HOME/$STAGE_FOLDER

source /opt/conda/bin/activate
conda activate cubedgx2

# step 1, only run on master node
max_retries=3
attempt_num=1

while [ $attempt_num -le $max_retries ]; do
    echo "Attempt #$attempt_num: Executing graph generation..."
    bash scripts/generalist/cube/run_stage_x.sh trace

    if [ -f "graph.cube" ]; then
        echo "Graph generated successfully."
        break
    else
        echo "Graph generation failed."
    fi

    if [ $attempt_num -eq $max_retries ]; then
        echo "Reached maximum number of retries, exiting."
        exit 1
    fi

    ((attempt_num++))

    sleep 5
done

# step 2, only run on master node
bash scripts/generalist/cube/run_stage_x.sh compile
if [ -f "gencode0.py" ]; then
    echo "Code generated successfully."
else
    echo "Code generation failed."
    exit 1
fi

cp SFM_framework/sfm/utils/barrier.py . && touch READY && python -u barrier.py $OMPI_COMM_WORLD_SIZE $OMPI_COMM_WORLD_RANK

# step 3, run on all node
bash scripts/generalist/cube/run_stage_x.sh run

# step 4 & 5, run on master node
bash scripts/generalist/cube/run_stage_x.sh mergeckpt checkpoint/checkpoint_last-shard0.pt
bash scripts/generalist/cube/run_stage_x.sh extract_model checkpoint/checkpoint_last-full.pt
