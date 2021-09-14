#!/usr/bin/env bash
set -x

export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=eth
export FI_PROVIDER="efa"
export FI_EFA_TX_MIN_CREDITS=64
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export NCCL_MIN_NRINGS=1
export NCCL_TREE_THRESHOLD=4294967296
export OMP_NUM_THREADS=8
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SOCKET_NTHREADS=8
export NCCL_BUFFSIZE=2097152

# sh run.sh 8 1 0 10.0.87.143 11122 1 eth0
NPROC_PER_NODE=$1
NNODE=$2
NODE_RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5
IB=$6
IF_NAME=$7

python3 -m launch \
--nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODE --node_rank=$NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
test_rpc_pipeline.py \
--nnodes $NNODE \
--nproc_per_node=$NPROC_PER_NODE \
--node_rank $NODE_RANK \
--is_infiniband $IB \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
--if_name $IF_NAME