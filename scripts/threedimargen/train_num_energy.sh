#python
torchrun --nnodes 1 --nproc_per_node 1 sfm/tasks/threedimargen/train_threedimargennumenergy.py \
    --save_dir /hai1/SFM/threedimargen/outputs/3darenergy_v0.4_2500k_scal10_ddp_layer24_head16_epoch100_warmup8000_lr1e-4_bs32 \
    --train_data_path /hai1/SFM/threedimargen/data/materials_data/coredataset-v20230731_train.jsonl \
    --valid_data_path /hai1/SFM/threedimargen/data/materials_data/coredataset-v20230731_test.jsonl \
    --fp16 \
    --total_num_epochs 100 \
    --warmup_num_steps 8000 \
    --max_lr 1e-4 \
    --train_batch_size 32 \
    --val_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --log_interval 100 \
    --scale_digit 10 \
    --strategy DDP \
