NAME=3dargenlan_v0.1_base_ft_mp_20_ddp_noniggli_base_epoch10_warmup1_lr1e-5_wd0.1_bs16

python sfm/tasks/threedimargen/train_threedimargenlan.py \
    --dict_path sfm/data/threedimargen_data/dict_lan.txt \
    --train_data_path /hai1/SFM/threedimargen/data/materials_data/mp_20_train.jsonl \
    --valid_data_path /hai1/SFM/threedimargen/data/materials_data/mp_20_val.jsonl \
    --save_dir /hai1/SFM/threedimargen/outputs/${NAME} \
    --model_type threedimargen \
    --tokenizer lan \
    --ft \
    --finetune_from_checkpoint_dir /hai1/SFM/threedimargen/outputs/3dargenlan_v0.1_base_mp_nomad_qmdb_ddp_noniggli_layer24_head16_epoch50_warmup8000_lr1e-4_wd0.1_bs256 \
    --finetune_from_checkpoint_id checkpoint_E40.pt \
    --fp16 \
    --total_num_epochs 10 \
    --warmup_num_steps 1 \
    --max_lr 1e-5 \
    --weight_decay 0.1 \
    --train_batch_size 16 \
    --val_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --log_interval 100 \
    #--niggli_reduced \
