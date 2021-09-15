import logging
import string
from typing import Any, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch import nn

# Based on https://github.com/pytorch/pytorch/pull/40762
from .layers import LinearActivation


# d_model -> hidden_size
# d_ff -> intermediate_size


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


class BertDenseExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, hidden_dropout_prob):
        super(BertDenseExpert, self).__init__()

        # intermediate Dense
        self.dense_act = LinearActivation(hidden_size, intermediate_size, act=hidden_act)

        # output Dense
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dense.bert_output_layer = True
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states):
        #        logging.debug("\n  BertDenseExpert_1:{}\n".format(hidden_states.shape))
        # intermediate dense
        hidden_states = self.dense_act(hidden_states)
        #        logging.debug("\n  BertDenseExpert_2:{}\n".format(hidden_states.shape))
        # output dense
        hidden_states = self.dense(hidden_states)
        #        logging.debug("\n  BertDenseExpert_3:{}\n".format(hidden_states.shape))
        # Iman hack 1. M5 code applies dropout here. But Switch applies dropout after experts.
        # hidden_states = self.dropout(hidden_states)

        return hidden_states


class ExpertParallelSwitchTransformerBase(nn.Module):
    def __init__(
        self,
        process_group,
        process_group_size: int,
        capacity_factor: float,
        drop_tokens: bool,
        sequence_length: int,
        global_rank: int,
    ):
        super().__init__()
        self.process_group = process_group
        self.n_experts = process_group_size
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.sequence_length = sequence_length

        self.global_rank = global_rank

    def Switch(self, input_):
        raise NotImplementedError()

    def Shuffle(self, input_):
        raise NotImplementedError()

    def Expert(self, input_):
        raise NotImplementedError()

    def Dropout(self, input_):
        raise NotImplementedError()

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
        route_prob_max, indexes_list, x = self.Switch(x)
        logging.debug("global_rank = {}, route_prob_max = {}".format(global_rank, route_prob_max))

        # Initialize an empty tensor to store outputs
        final_output = x.new_zeros(x.shape)

        # Capacity of each expert.
        # $$\mathrm{expert\;capacity} =
        # \frac{\mathrm{tokens\;per\;batch}}{\mathrm{number\;of\;experts}}
        # \times \mathrm{capacity\;factor}$$
        sequence_length = self.sequence_length
        capacity = int(
            self.capacity_factor * batch_size * sequence_length / self.n_experts
        )  ## hack hack , kalkarth fix seq length alignment properly.

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
        final_output = final_output.view(batch_size, seq_len, d_model)
        logging.debug(
            "global_rank = {}, final_output.shape = {}, final_output = {}".format(
                global_rank, final_output.shape, final_output
            )
        )

        final_output = self.Dropout(final_output)

        self.router_lb_loss = RouterLoadBalancingLoss()
        self.router_lb_loss.add_num_dropped_token(len(dropped))
        return final_output


class ExpertParallelSwitchFeedForward(ExpertParallelSwitchTransformerBase):
    """
    ## Routing among multiple FFNs (local + remote)
    """

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
        """
        * `capacity_factor` is the capacity of each expert as a factor relative to ideally balanced load
        * `drop_tokens` specifies whether to drop tokens if more tokens are routed to an expert than the capacity
        * `is_scale_prob` specifies whether to multiply the input to the FFN by the routing probability
        * `n_experts` is the number of experts
        * `expert` is the expert layer, a [FFN module](../feed_forward.html)
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability in the FFN
        """
        super().__init__(
            process_group,
            process_group_size,
            capacity_factor,
            drop_tokens,
            sequence_length,
            global_rank,
        )

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

        # Iman hack. moved dropout here
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


class RouterLoadBalancingLoss:
    """
    ## RouterLoadBalancingLoss is a Singleton class to collect losses from all Transformer FFN layers.
    the unit test code for this class is maintained at:
    test/switch_transformer/test_switch_transformer_layer.py
    """

    _instance = None

    def __new__(self, *args, **kw):
        # accumulated losses from all FFN layers
        self.total_lb_loss = None
        self.last_lb_loss = None
        self.num_dropped_token = 0

        if self._instance is None:
            self._instance = object.__new__(self, *args, **kw)
        return self._instance

    def calculate_and_accumulate_lb_loss(self, num_experts, router_probs, route_index):
        # logging.debug("router_probs = {}".format(router_probs))
        # logging.debug("route_index = {}".format(route_index))
        num_tokens = router_probs.size()[0]
        # logging.debug("num_tokens = {}".format(num_tokens))
        router_probs_per_expert = torch.sum(router_probs, dim=0)
        # logging.debug("router_probs_per_expert = {}".format(router_probs_per_expert))
        num_token_per_expert = torch.bincount(route_index, minlength=num_experts)
        # logging.debug("num_token_per_expert = {}".format(num_token_per_expert))

        # Equation (4) in Switch Transformer: https://arxiv.org/pdf/2101.03961.pdf
        loss = (
            torch.sum((num_token_per_expert / num_tokens) * (router_probs_per_expert / num_tokens))
            * num_experts
        )
        self.last_lb_loss = loss
        # logging.info("global_rank = {}, load_balancing_loss = {}".format(global_rank, loss))

        # we need to sum up the loss from all routers in different Transformer FFN layers.
        if self.total_lb_loss is None:
            self.total_lb_loss = loss
        else:
            torch.add(self.total_lb_loss, loss)
        return loss

    def get_last_lb_loss(self):
        return self.last_lb_loss

    def get_total_lb_loss(self):
        return self.total_lb_loss

    def clear_loss(self):
        self.total_lb_loss = None
        self.total_lb_loss = None
        self.num_dropped_token = 0

    def add_num_dropped_token(self, num_dropped_token):
        self.num_dropped_token += num_dropped_token

    def get_num_dropped_token(self):
        return self.num_dropped_token
