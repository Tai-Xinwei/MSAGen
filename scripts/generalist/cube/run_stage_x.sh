##################################################
## torchrun related
# number of devices per node, for DGX2, it is 16

TORCH_RUN_CMD="--nproc_per_node=$NPROC_PER_NODE \
--nnodes=$NNODES \
--node_rank=$NODE_RANK \
--master_addr=$MASTER_ADDR \
--master_port=$MASTER_PORT "

##################################################
## file path & data path
SCRIPT_PATH="$(dirname "$(readlink -f "$0")")"

OUTPUT_DIR=$SAVE_DIR
TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard
LOG_DIR=$OUTPUT_DIR/log
CHECKPOINT_DIR=$OUTPUT_DIR/checkpoint

mkdir -p $TENSORBOARD_DIR
mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR

##################################################
## model & data related
TASK=ai4sci
ARCH=graphormer_llama

##################################################

MODEL_PARAMS="--arch $ARCH \
--llm-model-name-or-path $LLAMA_CKPT_PATH \
--model-train-stage $STAGE \
--graphormer-ckpt-path $GRAPHORMER_CKPT_PATH \
"

if [ -n "$FULL_MODEL_CKPT_PATH" ]; then
    MODEL_PARAMS="$MODEL_PARAMS --full-model-ckpt-path $FULL_MODEL_CKPT_PATH"
fi

TASK_PARAMS="--task $TASK \
--criterion ai4sci_copilot \
--validate-interval-updates -1 \
--disable-validation \
--save-interval-updates 1000 \
--log-interval 1 \
--pad-to-fixed-length \
--tokens-per-sample $TOKENS_PER_SAMPLE \
--max-num-mol-per-sample $MAX_NUM_MOL_PER_SAMPLE \
--batch-size $BATCH_SIZE \
--required-batch-size-multiple 1 \
--update-freq $UPDATE_FREQ \
--warmup-updates $WARMUP_UPDATES \
--max-update $MAX_UPDATES \
--seed $SEED \
$MODEL_PARAMS"

DATA_PARAMS="--data-path $DATA_PATH \
--dataset-names $DATASET_NAMES \
--dataset-splits $DATASET_SPLITS \
--dataset-ratios $DATASET_RATIOS
"

##################################################
## optimize related
OPTIMIZE_PARAMS="
--optimizer adam \
--adam-eps 1e-08 \
--clip-norm 1.0 \
--lr 2e-4 \
--lr-scheduler polynomial_decay \
--total-num-update $MAX_UPDATES \
--weight-decay 0.0 \
--fp16 \
--memory-efficient-fp16 \
--fp16-init-scale 1 "

##################################################
## cube related

# in cube
# - all devices are divided into several scale units
# - data parallelism is applied between scale units
# - each scale unit has the same number of devices
# - each scale unit has a batch size of BATCH_SIZE
# - the total batch size is BATCH_SIZE * SCALE_UNIT_NUM.

GRAPH_PATH="graph.cube"
PLAN_NGPUS=$NPROC_PER_NODE
CUBE_SCALING_FACTOR=$NNODES

USE_ZERO=1
ZERO_GROUP_SIZE=$(($NPROC_PER_NODE * $NNODES / $ZERO_N_GROUPS))
ASYNC_REDUCER=0

# memory constraint in GB for distributed plan searching in autodist
AUTODIST_MEM_PER_GPU=24

CUBE_ENV="env PLAN_NGPUS=$PLAN_NGPUS USE_ZERO=$USE_ZERO ASYNC_REDUCER=$ASYNC_REDUCER ZERO_NUM_GROUPS=$ZERO_N_GROUPS"

CUBE_PARAMS="--ddp-backend legacy_ddp \
--cube-scaling-factor $CUBE_SCALING_FACTOR \
--parallel-backend cube \
--tensorboard-logdir $TENSORBOARD_DIR \
--save-dir $CHECKPOINT_DIR "

##################################################
## main logic

if [ $# -lt 1 ]
then
    echo "Usage: bash run.sh <mode> <args>"
    echo "       mode = {trace, compile, run}"
    exit 1
fi

MODE=$1

if [ $MODE = "trace" ]
then
    DATA_PARAMS="--data-path $DATA_PATH --dataset-names chebi --dataset-splits all --dataset-ratios 1.0"
    $CUBE_ENV torchrun --nproc_per_node=1 --nnodes=1 $EXAMPLE_PATH/train.py $TASK_PARAMS $DATA_PARAMS --num-workers 0 $OPTIMIZE_PARAMS $CUBE_PARAMS --adam-betas "(0.9,0.999)" --compile=compile_only --trace-only --cube-save-graph-ckp $GRAPH_PATH >$LOG_DIR/trace_log_${NODE_RANK}.txt 2>&1
elif [ $MODE = "compile" ]
then
    DATA_PARAMS="--data-path $DATA_PATH --dataset-names chebi --dataset-splits all --dataset-ratios 1.0"
    $CUBE_ENV torchrun --nproc_per_node=1 --nnodes=1 $EXAMPLE_PATH/train.py $TASK_PARAMS $DATA_PARAMS --num-workers 0 $OPTIMIZE_PARAMS $CUBE_PARAMS --adam-betas "(0.9,0.999)" --compile=compile_only --enable-autodist --autodist-verbose --mesh-row 1 --mesh-col $PLAN_NGPUS --autodist-mem-constraint $AUTODIST_MEM_PER_GPU --cube-load-graph-ckp $GRAPH_PATH >$LOG_DIR/autodist_log_${NODE_RANK}.txt 2>&1
elif [ $MODE = "run" ]
then
    $CUBE_ENV torchrun $TORCH_RUN_CMD $EXAMPLE_PATH/train.py $TASK_PARAMS $DATA_PARAMS --num-workers $NUM_WORKER $OPTIMIZE_PARAMS $CUBE_PARAMS --adam-betas "(0.9,0.999)" --compile=run_only  --cube-load-graph-ckp $GRAPH_PATH >$LOG_DIR/run_log_${NODE_RANK}.txt 2>&1
elif [ $MODE = "mergeckpt" ]
then
    if [ $# -lt 2 ]
    then
        echo "Usage: bash run.sh mergeckpt <checkpoint_path>"
        exit 1
    fi

    # e.g., ./checkpoints/checkpoint_last-shard0.pt
    CKPTPATH=$2
    # check "-shard0.pt" is suffix of CKPTPATH
    if [[ $CKPTPATH != *"-shard0.pt" ]]; then
        echo "please specify the checkpoint file with suffix -shard0.pt"
        exit 1
    fi

    $CUBE_ENV python -c "from fairseq.cube.cube_trainer import CubeTrainer; CubeTrainer.merge_checkpoints('$CKPTPATH', $ZERO_GROUP_SIZE)"

    # replace the "-shard0.pt" suffix with "-full.pt" in CKPTPATH
    MERGED_CKPTPATH=${CKPTPATH/-shard0.pt/-full.pt}
    echo "Created the merged checkpoint file $MERGED_CKPTPATH, please place it in blob and specify this file in the run command to resume."
elif [ $MODE = "extract_model" ]
then
    CKPTPATH=$2
    if [[ $CKPTPATH != *"-full.pt" ]]; then
        echo "please specify the checkpoint file with suffix -full.pt"
        exit 1
    fi
    python $EXAMPLE_PATH/extract_state_dict.py $CKPTPATH
fi
