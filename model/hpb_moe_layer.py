import logging
import string

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.pipeline.sync import Pipe
from torch.distributed.pipeline.sync.skip import skippable, stash, pop

from .switch_transformer_layers import (
    RouterLoadBalancingLoss,
    BertDenseExpert,
)
from .switch_transformer_layers import _AllToAll


class IntraNodeMoELayer(nn.Module):
    def __init__(
        self,
        process_group,
        process_group_size: int,
        global_rank: int,
        capacity_factor: float,
        drop_tokens: bool,
        sequence_length: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: string,
        hidden_dropout_prob: float,
    ):
        super().__init__()
        self.process_group = process_group
        self.n_experts = process_group_size
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.sequence_length = sequence_length

        self.global_rank = global_rank

        # d_model = hidden_size
        d_model = hidden_size

        # Each device has one expert.
        self.local_expert = BertDenseExpert(
            hidden_size, intermediate_size, hidden_act, hidden_dropout_prob
        )

        # Routing layer and softmax
        self.switch = nn.Linear(d_model, self.n_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.router_lb_loss = RouterLoadBalancingLoss()

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def Switch(self, x):
        """
        * `x` is the input to the switching module with shape `[seq_len, batch_size, d_model]`
        """
        # Get routing probabilities for each of the tokens.
        # $$p_i(x) = \frac{e^{h(x)_i}}{\sum^N_j e^{h(x)_j}}$$
        # where $N$ is the number of experts `n_experts` and
        # $h(\cdot)$ is the linear transformation of token embeddings.
        route_prob = self.softmax(self.switch(x))

        # Get the maximum routing probabilities and the routes.
        # We route to the expert with highest probability
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        self.router_lb_loss.calculate_and_accumulate_lb_loss(self.n_experts, route_prob, routes)

        # Get indexes of tokens going to each expert
        indexes_list = [
            torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)
        ]
        return route_prob_max, indexes_list, x

    def Expert(self, x):
        return self.local_expert(x)

    def Dropout(self, x):
        return self.dropout(x)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input to the switching module with shape `[seq_len, batch_size, d_model]`
        """
        # expert_outputs = self.Expert(torch.cat(route_inputs))
        global_rank = self.global_rank
        logging.debug("global_rank = %d" % global_rank)

        intra_node_seq_len, d_model = x.shape
        logging.debug(
            "global_rank = {}, intra_node_seq_len = {}".format(global_rank, intra_node_seq_len)
        )

        # Perform switch to get index lists
        route_prob_max, indexes_list, x = self.Switch(x)
        logging.debug("global_rank = {}, route_prob_max = {}".format(global_rank, route_prob_max))

        # Initialize an empty tensor to store outputs
        final_output = x.new_zeros(x.shape)

        capacity = int(self.capacity_factor * intra_node_seq_len / self.n_experts)

        # Number of tokens routed to each expert.
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        # Initialize an empty list of dropped tokens
        dropped = []
        # Only drop tokens if `drop_tokens` is `True`.
        logging.debug(
            "(before drop) global_rank = {}, indexes_list = {}".format(global_rank, indexes_list)
        )
        if self.drop_tokens:
            # Drop tokens in each of the experts
            for i in range(self.n_experts):
                # Ignore if the expert is not over capacity
                if len(indexes_list[i]) <= capacity:
                    continue
                # Shuffle indexes before dropping
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                # Collect the tokens over capacity as dropped tokens
                dropped.append(indexes_list[i][capacity:])
                # Keep only the tokens upto the capacity of the expert
                indexes_list[i] = indexes_list[i][:capacity]

        # Get inputs of expert FNNs
        logging.debug(
            "(after drop)global_rank = {}, indexes_list = {}".format(global_rank, indexes_list)
        )
        route_inputs = [x[indexes_list[i], :] for i in range(self.n_experts)]
        for i in range(self.n_experts):
            # Ignore if the expert is not over capacity
            if len(indexes_list[i]) < capacity:
                # Pad to capacity.
                # (TODO:kalkarth) Find right pad value.
                route_inputs[i] = F.pad(
                    route_inputs[i],
                    (0, 0, 0, capacity - len(indexes_list[i])),
                    "constant",
                    value=0,
                )

        # (all_to_all) Shuffle inputs to right experts
        logging.debug(
            "(before all_to_all 1, global_rank = {}) route_inputs = {}".format(
                global_rank, route_inputs
            )
        )
        expert_inputs = _AllToAll.apply(self.process_group, torch.cat(route_inputs))
        logging.debug(
            "(after all_to_all 1, global_rank = {}) expert_inputs.shape = {}, expert_inputs = {}".format(
                global_rank, expert_inputs.shape, expert_inputs
            )
        )

        # Evaluate experts
        # logging.debug("self.Expert parameters:")
        # for name, param in self.local_expert.named_parameters():
        #     if param.requires_grad:
        #         logging.debug(
        #             "globak_rank = {}, name = {}, param.data = {}".format(
        #                 global_rank, name, param.data
        #             )
        #         )
        expert_output = self.local_expert(expert_inputs)

        # (all_to_all) Shuffle outputs to right sources
        logging.debug(
            "(before all_to_all 2, global_rank = {}) expert_output.shape = {}, expert_output = {}".format(
                global_rank, expert_output.shape, expert_output
            )
        )
        expert_outputs = _AllToAll.apply(self.process_group, expert_output)
        logging.debug(
            "(after all_to_all 2, global_rank = {}) expert_outputs.shape = {}, expert_outputs = {}".format(
                global_rank, expert_outputs.shape, expert_outputs
            )
        )

        assert expert_outputs.size(0) == self.n_experts * capacity, "{} is not  {}".format(
            expert_outputs.size(0), x.size(0)
        )

        # Assign to final output
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_outputs[
                i * capacity : i * capacity + indexes_list[i].size(0)
            ]

        logging.debug(
            "global_rank = {}, before adding dropped tokens, final_output.shape = {}, final_output = {}".format(
                global_rank, final_output.shape, final_output
            )
        )

        # Pass through the dropped tokens
        route_prob_factor = route_prob_max
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

            """
            If too many tokens are routed to an expert (referred to later as dropped tokens), 
            computation is skipped and the token representation is passed directly to the next layer through the residual connection
            """
            route_prob_factor[dropped] = 1.0

        logging.debug(
            "global_rank = {}, after adding dropped tokens, final_output.shape = {}, final_output = {}".format(
                global_rank, final_output.shape, final_output
            )
        )

        # Scale the output of experts by the routing probabilities
        final_output = final_output * route_prob_factor.view(-1, 1)
        logging.debug(
            "(routing probabilities) global_rank = {}, final_output.shape = {}, final_output = {}".format(
                global_rank, final_output.shape, final_output
            )
        )

        # Change the shape of the final output back to `[seq_len, batch_size, d_model]`
        # final_output = final_output.view(seq_len, batch_size, d_model)
        final_output = final_output.view(intra_node_seq_len, d_model)
        logging.debug(
            "global_rank = {}, final_output.shape = {}, final_output = {}".format(
                global_rank, final_output.shape, final_output
            )
        )

        final_output = self.Dropout(final_output)

        self.router_lb_loss = RouterLoadBalancingLoss()
        self.router_lb_loss.add_num_dropped_token(len(dropped))

        # (all_to_all) Shuffle outputs to right sources
        logging.debug(
            "(before all_to_all 2, global_rank = {}) output_of_intra_node_moe_tensor.shape ={}, "
            "output_of_intra_node_moe_tensor = {}".format(
                global_rank, final_output.shape, final_output
            )
        )
        return final_output


class InterNodeMoELayerIn(nn.Module):
    def __init__(
        self,
        shared_module,
        inter_node_process_group,
        inter_node_process_group_size: int,
        global_rank: int,
        capacity_factor: float,
        drop_tokens: bool,
        sequence_length: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: string,
        hidden_dropout_prob: float,
    ):
        # (inter-node) Routing layer and softmax
        super().__init__()
        self.shared_module = shared_module
        self.global_rank = global_rank
        self.sequence_length = sequence_length
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens

        self.inter_node_process_group = inter_node_process_group
        self.n_experts_inter = inter_node_process_group_size
        self.switch_inter = nn.Linear(hidden_size, self.n_experts_inter)
        self.softmax = nn.Softmax(dim=-1)

    def Switch_inter(self, x):
        """
        * `x` is the input to the switching module with shape `[seq_len, batch_size, d_model]`
        """
        # Get routing probabilities for each of the tokens.
        # $$p_i(x) = \frac{e^{h(x)_i}}{\sum^N_j e^{h(x)_j}}$$
        # where $N$ is the number of experts `n_experts` and
        # $h(\cdot)$ is the linear transformation of token embeddings.
        route_prob = self.softmax(self.switch_inter(x))

        # Get the maximum routing probabilities and the routes.
        # We route to the expert with highest probability
        route_prob_max, routes = torch.max(route_prob, dim=-1)

        # Get indexes of tokens going to each expert
        indexes_list = [
            torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts_inter)
        ]
        return route_prob_max, indexes_list, x

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input to the switching module with shape `[seq_len, batch_size, d_model]`
        """
        # expert_outputs = self.Expert(torch.cat(route_inputs))
        global_rank = self.global_rank
        logging.debug("global_rank = %d" % global_rank)

        # iman hack. M5 shape is different. So I changed the code in two places. below and ad the end
        # Capture the shape to change shapes later
        # seq_len, batch_size, d_model = x.shape
        batch_size, seq_len, d_model = x.shape  # M5 shape
        # Flatten the sequence and batch dimensions
        x = x.view(-1, d_model)

        # assert batch_size == 32, "Unexpected batch size {} in rank {}".format(batch_size, torch.distributed.get_rank())
        # assert seq_len == 128, "Unexpected sequence size {} in rank {}".format(seq_len, torch.distributed.get_rank())

        # Perform switch to get index lists
        route_prob_max, indexes_list, x = self.Switch_inter(x)
        logging.debug("global_rank = {}, route_prob_max = {}".format(global_rank, route_prob_max))

        # batch_size = 32
        # hidden_size = 1024
        # intermediate_size = 4096
        # hidden_dropout_prob = 0.1
        # sequence_length = 128
        sequence_length = self.sequence_length
        logging.debug(
            "capacity_factor = {}, batch_size = {}, sequence_length = {}, n_experts_inter = {}".format(
                self.capacity_factor, batch_size, sequence_length, self.n_experts_inter
            )
        )
        capacity = int(self.capacity_factor * batch_size * sequence_length / self.n_experts_inter)
        logging.debug("global_rank = {}, capacity = {}".format(global_rank, capacity))

        # Number of tokens routed to each expert.
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts_inter)])

        # Initialize an empty list of dropped tokens
        dropped = []
        # Only drop tokens if `drop_tokens` is `True`.
        logging.debug(
            "(before drop) global_rank = {}, indexes_list = {}".format(global_rank, indexes_list)
        )
        if self.drop_tokens:
            # Drop tokens in each of the experts
            for i in range(self.n_experts_inter):
                # Ignore if the expert is not over capacity
                if len(indexes_list[i]) <= capacity:
                    continue
                # Shuffle indexes before dropping
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                # Collect the tokens over capacity as dropped tokens
                dropped.append(indexes_list[i][capacity:])
                # Keep only the tokens upto the capacity of the expert
                indexes_list[i] = indexes_list[i][:capacity]

        # Get inputs of expert FNNs
        logging.debug(
            "(after drop)global_rank = {}, indexes_list = {}".format(global_rank, indexes_list)
        )
        route_inputs = [x[indexes_list[i], :] for i in range(self.n_experts_inter)]
        for i in range(self.n_experts_inter):
            # Ignore if the expert is not over capacity
            if len(indexes_list[i]) < capacity:
                # Pad to capacity.
                # (TODO:kalkarth) Find right pad value.
                route_inputs[i] = F.pad(
                    route_inputs[i],
                    (0, 0, 0, capacity - len(indexes_list[i])),
                    "constant",
                    value=0,
                )

        # (all_to_all) Shuffle inputs to right experts
        logging.debug(
            "(before all_to_all 1, global_rank = {}) route_inputs = {}".format(
                global_rank, route_inputs
            )
        )
        inputs_for_intra_node_moe_tensor = _AllToAll.apply(
            self.inter_node_process_group, torch.cat(route_inputs)
        )
        # [24, 3]
        logging.debug(
            "(after all_to_all 1, global_rank = {}) inputs_for_intra_node_moe_tensor.shape = {}, "
            "inputs_for_intra_node_moe_tensor = {}".format(
                global_rank,
                inputs_for_intra_node_moe_tensor.shape,
                inputs_for_intra_node_moe_tensor,
            )
        )

        self.shared_module.update_shared_params(
            x.view(batch_size, seq_len, d_model), route_prob_max, indexes_list, dropped, capacity
        )
        return inputs_for_intra_node_moe_tensor


class InterNodeMoELayerOut(nn.Module):
    def __init__(
        self,
        shared_module,
        inter_node_process_group,
        inter_node_process_group_size: int,
        global_rank: int,
        capacity_factor: float,
        drop_tokens: bool,
        sequence_length: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: string,
        hidden_dropout_prob: float,
    ):
        super().__init__()
        self.shared_module = shared_module
        self.inter_node_process_group = inter_node_process_group
        self.global_rank = global_rank
        self.n_experts_inter = inter_node_process_group_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.intermediate_size = intermediate_size
        self.drop_tokens = drop_tokens

        self.dropout_inter = nn.Dropout(hidden_dropout_prob)

    def forward(self, output_of_intra_node_moe_tensor):
        logging.debug(
            "inter-node out output_of_intra_node_moe_tensor.shape = {}, output_of_intra_node_moe_tensor = {}".format(
                output_of_intra_node_moe_tensor.shape, output_of_intra_node_moe_tensor
            )
        )

        x, route_prob_max, indexes_list, dropped, capacity = self.shared_module.get_shared_params()

        batch_size, seq_len, d_model = x.shape  # M5 shape
        # Flatten the sequence and batch dimensions
        x = x.view(-1, d_model)

        # Initialize an empty tensor to store outputs
        final_output = x.new_zeros(x.shape)

        seq_len = self.sequence_length
        d_model = self.hidden_size

        global_rank = self.global_rank

        logging.info(
            "(before all_to_all 2, global_rank = {}) output_of_intra_node_moe_tensor.shape = {}, "
            "output_of_intra_node_moe_tensor = {}".format(
                global_rank,
                output_of_intra_node_moe_tensor.shape,
                output_of_intra_node_moe_tensor,
            )
        )
        outputs_of_intra_node_moe_tensor = _AllToAll.apply(
            self.inter_node_process_group, output_of_intra_node_moe_tensor
        )
        logging.debug(
            "(after all_to_all 2, global_rank = {}) outputs_of_intra_node_moe_tensor.shape = {}, "
            "outputs_of_intra_node_moe_tensor = {}".format(
                global_rank,
                outputs_of_intra_node_moe_tensor.shape,
                outputs_of_intra_node_moe_tensor,
            )
        )

        assert (
            outputs_of_intra_node_moe_tensor.size(0) == self.n_experts_inter * capacity
        ), "{} is not  {}".format(outputs_of_intra_node_moe_tensor.size(0), x.size(0))

        # Assign to final output
        logging.info("final_output = {}".format(final_output))
        logging.info("indexes_list = {}".format(indexes_list))
        logging.info("capacity = {}".format(capacity))
        for i in range(self.n_experts_inter):
            final_output[indexes_list[i], :] = outputs_of_intra_node_moe_tensor[
                i * capacity : i * capacity + indexes_list[i].size(0)
            ]

        logging.debug(
            "global_rank = {}, before adding dropped tokens, final_output.shape = {}, final_output = {}".format(
                global_rank, final_output.shape, final_output
            )
        )

        # Pass through the dropped tokens
        route_prob_factor = route_prob_max
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

            """
            If too many tokens are routed to an expert (referred to later as dropped tokens), 
            computation is skipped and the token representation is passed directly to the next layer through the residual connection
            """
            route_prob_factor[dropped] = 1.0

        logging.debug(
            "global_rank = {}, after adding dropped tokens, final_output.shape = {}, final_output = {}".format(
                global_rank, final_output.shape, final_output
            )
        )

        # Scale the output of experts by the routing probabilities
        final_output = final_output * route_prob_factor.view(-1, 1)
        logging.debug(
            "(routing probabilities) global_rank = {}, final_output.shape = {}, final_output = {}".format(
                global_rank, final_output.shape, final_output
            )
        )

        # Change the shape of the final output back to `[seq_len, batch_size, d_model]`
        # final_output = final_output.view(seq_len, batch_size, d_model)
        final_output = final_output.view(batch_size, seq_len, d_model)
        logging.debug(
            "global_rank = {}, final_output.shape = {}, final_output = {}".format(
                global_rank, final_output.shape, final_output
            )
        )

        final_output = self.dropout_inter(final_output)

        self.router_lb_loss = RouterLoadBalancingLoss()
        self.router_lb_loss.add_num_dropped_token(len(dropped))
        return final_output


class PipeModelWrapper(nn.Module):
    def __init__(self, pipe_model):
        super().__init__()
        self.pipe_model = pipe_model

    def forward(self, *args, **kwargs):
        return self.pipe_model(*args, **kwargs).local_value()


class SharedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.original_x = None
        self.route_prob_max = None
        self.indexes_list = None
        self.dropped = None
        self.capacity = None

    def update_shared_params(self, original_x, route_prob_max, indexes_list, dropped, capacity):
        self.original_x = original_x
        self.route_prob_max = route_prob_max
        self.indexes_list = indexes_list
        self.dropped = dropped
        self.capacity = capacity

    def get_shared_params(self):
        return self.original_x, self.route_prob_max, self.indexes_list, self.dropped, self.capacity


@skippable(stash=["1to3"])
class Layer1(nn.Module):
    def __init__(self, hidden_size, local_rank, pg):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size).cuda(local_rank)
        self.pg = pg

    def forward(self, input):
        # logging.info("(before alltoall) global_rank = {}, input = {}".format(global_rank, input))
        input = _AllToAll.apply(self.pg, input)
        # logging.info("(after alltoall) global_rank = {}, input = {}".format(global_rank, input))
        yield stash("1to3", input)
        return self.fc1(input)


class Layer2(nn.Module):
    def __init__(self, hidden_size, local_rank):
        super().__init__()
        self.fc2 = nn.Linear(hidden_size, hidden_size).cuda(local_rank)

    def forward(self, input):
        return self.fc2(input)


@skippable(pop=["1to3"])
class Layer3(nn.Module):
    def __init__(self, hidden_size, local_rank):
        super().__init__()
        self.fc3 = nn.Linear(hidden_size, hidden_size).cuda(local_rank)

    def forward(self, input):
        skip_1to3 = yield pop("1to3")
        return self.fc3(input) + skip_1to3
