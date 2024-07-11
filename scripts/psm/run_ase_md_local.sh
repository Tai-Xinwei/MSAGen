
export MODEL_CONFIG="PSM300M_V0.yaml"
export data_path="/home/v-xixianliu/work/relax_data/POSCAR"
export load_ckpt_path="/home/v-xixianliu/work/xixian/checkpoint/psm/checkpoint_E8_B31280.pt"

torchrun --nproc_per_node 1 --master_port 62348 sfm/tasks/psm/ase_md_psm.py \
        --config-name=$MODEL_CONFIG \
        psm_validation_mode=true \
        rescale_loss_with_std=true \
        dataset_split_raito=1.0 \
        clean_sample_ratio=1.0 \
        data_path=$data_path \
        loadcheck_path=$load_ckpt_path
