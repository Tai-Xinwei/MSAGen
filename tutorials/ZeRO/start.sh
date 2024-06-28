torchrun --nproc_per_node 4 tutorial.py --total_num_epochs 10 --strategy Zero1 --train_batch_size 32 --val_batch_size 32 --fp16 --seed 666666
