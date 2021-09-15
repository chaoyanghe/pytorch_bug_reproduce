import argparse
import logging
import os
from datetime import timedelta
from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch import nn
from torch.distributed import Backend, rpc
from torch.distributed.pipeline.sync import Pipe
from torch.distributed.pipeline.sync.skip import stash, skippable, pop
from torch.nn.parallel import DistributedDataParallel as DDP


# Based on https://github.com/pytorch/pytorch/pull/40762
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
        self._init_rpc_with_torchpipe()

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

        self.ddp_group = None

    def _init_rpc_with_process_group(self):
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
        logging.info("_init_rpc_with_process_group finished.")

    def _init_rpc_with_torchpipe(self):
        # https://github.com/pytorch/pytorch/issues/55615
        # [BC-Breaking][RFC] Retire ProcessGroup Backend for RPC #55615
        str_init_method = "tcp://" + str(self.master_addr) + ":10000"
        logging.info("str_init_method = {}".format(str_init_method))
        options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=16, rpc_timeout=20, init_method=str_init_method, _transports=["uv"]
        )
        rpc.init_rpc(
            "worker:" + str(self.global_rank),
            backend=rpc.BackendType.TENSORPIPE,
            rank=self.global_rank,
            world_size=self.world_size,
            rpc_backend_options=options,
        )
        # torch.distributed.rpc.init_rpc('worker', rank=self.global_rank, world_size=self.world_size)
        logging.info("_init_rpc_with_torchpipe finished.")

    def generate_ddp_model(self, model):
        # all_reduce group
        ddp_ranks = [rank for rank in range(self.world_size)]
        self.ddp_group = dist.new_group(
            ranks=ddp_ranks,
            backend=Backend.NCCL,
            timeout=timedelta(days=365),
        )
        model = DDP(model, process_group=self.ddp_group)
        return model

    def get_global_rank(self):
        return self.global_rank

    def get_lobal_rank(self):
        return self.local_rank

    def get_ddp_group(self):
        return self.ddp_group


class PipeModelWrapper(nn.Module):
    def __init__(self, pipe_model):
        super().__init__()
        self.pipe_model = pipe_model

    def forward(self, *args, **kwargs):
        return self.pipe_model(*args, **kwargs).local_value()


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


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

    batch_size = 16
    hidden_size = 2

    device = torch.device("cuda:" + str(local_rank))

    def prepare_training_data(batch_size, hidden_size):
        # logging.info(
        #     "batch_size = {}, max_seq_length = {}, hidden_size = {}".format(
        #         batch_size,
        #         max_seq_length,
        #         hidden_size,
        #     )
        # )
        x_dummpy = torch.rand(batch_size, hidden_size, hidden_size)
        return x_dummpy

    @skippable(stash=["1to3"])
    class Layer1(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, hidden_size).cuda(local_rank)

        def forward(self, input):
            # logging.info("(before alltoall) global_rank = {}, input = {}".format(global_rank, input))
            input = _AllToAll.apply(dist_mgr.get_ddp_group(), input)
            # logging.info("(after alltoall) global_rank = {}, input = {}".format(global_rank, input))
            yield stash("1to3", input)
            return self.fc1(input)

    class Layer2(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.fc2 = nn.Linear(hidden_size, hidden_size).cuda(local_rank)

        def forward(self, input):
            return self.fc2(input)

    @skippable(pop=["1to3"])
    class Layer3(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.fc3 = nn.Linear(hidden_size, hidden_size).cuda(local_rank)

        def forward(self, input):
            skip_1to3 = yield pop("1to3")
            return self.fc3(input) + skip_1to3

    class SuperModel(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()

            self.input_layer = nn.Linear(hidden_size, hidden_size)

            layer1 = Layer1(hidden_size)
            layer2 = Layer2(hidden_size)
            layer3 = Layer3(hidden_size)
            # Step 2: wrap the two layers with nn.Sequential
            pipeline_model = nn.Sequential(layer1, layer2, layer3)

            # Step 3: build Pipe (torch.distributed.pipeline.sync.Pipe)
            self.pipeline_model = PipeModelWrapper(
                Pipe(pipeline_model, chunks=2, checkpoint="never")
            )
            # self.pipeline_model = Pipe(pipeline_model, chunks=4, checkpoint="never")

            self.dense = nn.Linear(hidden_size, hidden_size)

        def forward(self, hidden_states):
            hidden_states = self.input_layer(hidden_states)
            pipeline_output = self.pipeline_model(hidden_states)
            output = self.dense(pipeline_output)
            return output

    super_model = SuperModel(hidden_size)
    super_model.to(device)

    super_model = dist_mgr.generate_ddp_model(super_model)
    if global_rank == 0:
        logging.info(super_model)

    ITER_NUM = 1
    super_model.train()
    time_cost_per_iter_total = 0.0
    for iter_idx in range(ITER_NUM):
        x = prepare_training_data(batch_size, hidden_size)
        x = x.to(device)
        output = super_model(x)

        # define a loss
        loss = torch.sum(output)
        loss.backward()

        if global_rank == 0:
            logging.info("loss = {}".format(loss))
