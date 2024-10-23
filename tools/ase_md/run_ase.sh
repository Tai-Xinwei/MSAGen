torchrun --nproc_per_node 1 --master_port 62349 \
         tools/ase_md/test_ase.py \
            --config-path . \
            --config-name final_config_chu \
            --steps 200 \
            --calculator pyscf \
            --name pyscf-200
