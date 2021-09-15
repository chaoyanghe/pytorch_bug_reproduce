from torch import nn
from torch.distributed.pipeline.sync import Pipe

from .hpb_moe_layer import (
    SharedModule,
    InterNodeMoELayerIn,
    InterNodeMoELayerOut,
    IntraNodeMoELayer,
)


class PipeModelWrapper(nn.Module):
    def __init__(self, pipe_model):
        super().__init__()
        self.pipe_model = pipe_model

    def forward(self, *args, **kwargs):
        return self.pipe_model(*args, **kwargs).local_value()


class TreeMoEModel(nn.Module):
    def __init__(
        self,
        intra_pg,
        intra_pg_size,
        inter_pg,
        inter_pg_size,
        global_rank,
        local_rank,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        sequence_length,
        pipe_chunk_size,
        b_open_pipeline,
    ):
        super().__init__()

        self.capacity_factor = 2.0
        self.drop_tokens = True
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = hidden_dropout_prob
        self.pipe_chunk_size = pipe_chunk_size

        self.global_rank = global_rank
        self.local_rank = local_rank

        self.n_experts_inter = inter_pg_size
        self.n_experts_intra = intra_pg_size
        self.inter_node_process_group = inter_pg
        self.intra_node_process_group = intra_pg
        self.intra_node_process_group_size = intra_pg_size
        self.inter_node_process_group_size = inter_pg_size

        self.b_open_pipeline = b_open_pipeline

        if self.b_open_pipeline:
            self.shared_module = SharedModule().cuda(self.local_rank)

            # (inter-node) Routing layer and softmax
            self.inter_node_moe_layer_in = InterNodeMoELayerIn(
                shared_module=self.shared_module,
                inter_node_process_group=self.inter_node_process_group,
                inter_node_process_group_size=self.inter_node_process_group_size,
                global_rank=self.global_rank,
                capacity_factor=self.capacity_factor,
                drop_tokens=self.drop_tokens,
                sequence_length=self.sequence_length,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                hidden_act=self.hidden_act,
                hidden_dropout_prob=self.hidden_dropout_prob,
            ).cuda(self.local_rank)

            self.inter_node_moe_layer_out = InterNodeMoELayerOut(
                shared_module=self.shared_module,
                inter_node_process_group=self.inter_node_process_group,
                inter_node_process_group_size=self.inter_node_process_group_size,
                global_rank=self.global_rank,
                capacity_factor=self.capacity_factor,
                drop_tokens=self.drop_tokens,
                sequence_length=self.sequence_length,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                hidden_act=self.hidden_act,
                hidden_dropout_prob=self.hidden_dropout_prob,
            ).cuda(self.local_rank)

            # (intra-node) MoE Layer
            self.intra_node_moe_layer = IntraNodeMoELayer(
                process_group=self.intra_node_process_group,
                process_group_size=self.intra_node_process_group_size,
                global_rank=self.global_rank,
                capacity_factor=self.capacity_factor,
                drop_tokens=self.drop_tokens,
                sequence_length=self.sequence_length,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                hidden_act=self.hidden_act,
                hidden_dropout_prob=self.hidden_dropout_prob,
            ).cuda(self.local_rank)

            self.dense = nn.Linear(hidden_size, hidden_size).cuda(self.local_rank)

            # Step 2: wrap the two layers with nn.Sequential
            pipeline_model = nn.Sequential(
                self.inter_node_moe_layer_in,
                self.intra_node_moe_layer,
                self.inter_node_moe_layer_out,
                self.dense,
            )

            # Step 3: build Pipe (torch.distributed.pipeline.sync.Pipe)
            self.pipeline_model = PipeModelWrapper(
                Pipe(pipeline_model, chunks=self.pipe_chunk_size, checkpoint="never")
            )
        else:
            raise Exception("no such implementation!")

    def forward(self, hidden_states):
        return self.pipeline_model(hidden_states)
