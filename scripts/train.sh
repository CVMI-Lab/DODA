set -x
NGPUS=$1
PY_FILE=$2
PY_ARGS=${@:3}

python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} tool/${PY_FILE}.py --launcher pytorch ${PY_ARGS}
