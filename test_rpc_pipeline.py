import argparse
import logging
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import Backend, rpc
from torch.distributed.pipeline.sync import Pipe
from torch.nn.parallel import DistributedDataParallel as DDP


def add_args():
    parser = argparse.ArgumentParser(description="MoE")

    parser.add_argument("--run_id", type=int, default=0)

    parser.add_argument("--nnodes", type=int, default=2)

    parser.add_argument("--nproc_per_node", type=int, default=8)

    parser.add_argument("--node_rank", type=int, default=0)

    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--global_rank", type=int, default=0)

    parser.add_argument("--world_size", type=int, default=0)

    parser.add_argument("--master_addr", type=str, default="192.168.11.2")

    parser.add_argument("--master_port", type=int, default=22222)

    parser.add_argument("--if_name", type=str, default="lo")

    parser.add_argument("--is_infiniband", default=1, type=int, help="is_infiniband")

    args = parser.parse_args()
    return args


class DistManager:
    def __init__(self, num_nodes, if_name, local_rank, master_addr, master_port):
        self.num_nodes = num_nodes
        self.if_name = if_name
        self.global_rank = -1
        self.local_rank = local_rank
        self.node_idx = -1
        self.world_size = -1

        self.master_addr = master_addr
        self.master_port = master_port

    def init(self):
        self._init_ddp()
        self._init_rpc()

    def _init_ddp(self):
        logging.info(f"Running DP on local rank {self.local_rank}.")

        self.master_port += 1
        os.environ.update({"MASTER_ADDR": self.master_addr})
        os.environ.update({"MASTER_PORT": str(self.master_port)})

        # use InfiniBand
        # os.environ['NCCL_DEBUG'] = 'INFO'

        os.environ["NCCL_SOCKET_IFNAME"] = self.if_name
        os.environ["GLOO_SOCKET_IFNAME"] = "eth0"  # only for init_rpc
        os.environ["TP_SOCKET_IFNAME"] = "eth0"

        # This the global rank: 0, 1, 2, ..., 15
        self.global_rank = int(os.environ["RANK"])
        logging.info("int(os.environ['RANK']) = %d" % self.global_rank)

        # This the globak world_size
        self.world_size = int(os.environ["WORLD_SIZE"])
        logging.info("world_size = %d" % self.world_size)

        # initialize the process group
        dist.init_process_group(
            init_method="tcp://" + str(self.master_addr) + ":" + str(self.master_port),
            backend=Backend.GLOO,
            rank=self.global_rank,
            world_size=self.world_size,
        )
        logging.info(
            "init_process_group. local_rank = %d, global_rank = %d"
            % (self.local_rank, self.global_rank)
        )

    def _init_rpc(self):
        # https://github.com/pytorch/pytorch/issues/55615
        # [BC-Breaking][RFC] Retire ProcessGroup Backend for RPC #55615
        str_init_method = "tcp://" + str(self.master_addr) + ":10000"
        logging.info("str_init_method = {}".format(str_init_method))
        options = rpc.ProcessGroupRpcBackendOptions(
            num_send_recv_threads=4, rpc_timeout=0.0, init_method=str_init_method
        )
        rpc.init_rpc(
            "worker:" + str(self.global_rank),
            backend=dist.rpc.BackendType.PROCESS_GROUP,
            rank=self.global_rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )
        # torch.distributed.rpc.init_rpc('worker', rank=self.global_rank, world_size=self.world_size)
        logging.info("init_rpc finished.")

    def generate_ddp_model(self, model):
        # all_reduce group
        ddp_ranks = [rank for rank in range(self.world_size)]
        ddp_group = dist.new_group(
            ranks=ddp_ranks,
            backend=Backend.NCCL,
            timeout=timedelta(days=365),
        )
        model = DDP(model, process_group=ddp_group)
        return model

    def get_global_rank(self):
        return self.global_rank

    def get_lobal_rank(self):
        return self.local_rank


class PipeModelWrapper(nn.Module):
    def __init__(self, pipe_model):
        super().__init__()
        self.pipe_model = pipe_model

    def forward(self, *args, **kwargs):
        return self.pipe_model(*args, **kwargs).local_value()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    args = add_args()

    dist_mgr = DistManager(
        num_nodes=args.nnodes,
        if_name=args.if_name,
        local_rank=args.local_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
    )
    dist_mgr.init()
    global_rank = dist_mgr.get_global_rank()
    local_rank = dist_mgr.get_lobal_rank()

    batch_size = 8
    max_seq_length = 2
    hidden_size = 4

    device = torch.device("cuda:" + str(local_rank))

    def prepare_training_data(batch_size, max_seq_length, hidden_size):
        # logging.info(
        #     "batch_size = {}, max_seq_length = {}, hidden_size = {}".format(
        #         batch_size,
        #         max_seq_length,
        #         hidden_size,
        #     )
        # )
        x_dummpy = torch.rand(batch_size, max_seq_length, hidden_size)
        return x_dummpy


    class SuperModel(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()

            # Step 1: build a model including two linear layers
            fc1 = nn.Linear(hidden_size, hidden_size).cuda(local_rank)
            fc2 = nn.Linear(hidden_size, hidden_size).cuda(local_rank)
            # Step 2: wrap the two layers with nn.Sequential
            pipeline_model = nn.Sequential(fc1, fc2)

            # Step 3: build Pipe (torch.distributed.pipeline.sync.Pipe)
            self.pipeline_model = PipeModelWrapper(Pipe(pipeline_model, chunks=4, checkpoint="never"))

            self.dense = nn.Linear(hidden_size, hidden_size)

        def forward(self, hidden_states):
            pipeline_output = self.pipeline_model(hidden_states)
            output = self.dense(pipeline_output)
            return output

    super_model = SuperModel(hidden_size)
    super_model.to(device)

    super_model = dist_mgr.generate_ddp_model(super_model)
    if global_rank == 0:
        logging.info(super_model)

    ITER_NUM = 10
    super_model.train()
    time_cost_per_iter_total = 0.0
    for iter_idx in range(ITER_NUM):
        x = prepare_training_data(batch_size, max_seq_length, hidden_size)
        x = x.to(device)
        output = super_model(x)

        # define a loss
        loss = torch.sum(output)
        loss.backward()

        if global_rank == 0:
            logging.info("loss = {}".format(loss))




