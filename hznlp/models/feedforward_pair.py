"""
A feed-forward neural network for pairs of input.
"""

import torch
import torch.nn as nn

from allennlp.nn import Activation
from allennlp.common.registrable import FromParams


class FeedForwardPair(nn.Module, FromParams):
    """
    This ``Module`` is a feed-forward neural network for pairs of input, it generates
    straight connections and cross connections.

    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 activation: Activation,
                 dropout: float = 0.0) -> None:

        super(FeedForwardPair, self).__init__()
        self.straight = nn.Linear(input_dim, hidden_dim)
        self.cross = nn.Linear(input_dim, hidden_dim)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        r1 = self.straight(input1) + self.cross(input2)
        r1 = self.dropout(self.activation(r1))
        r2 = self.straight(input2) + self.cross(input1)
        r2 = self.dropout(self.activation(r2))

        return r1, r2

if __name__ == "__main__":
    from allennlp.common import Params
    torch.manual_seed(999)

    batch = 16
    input_dim = 200
    output_dim = 100
    test_input_1 = torch.autograd.Variable(torch.randn(batch, input_dim))
    test_input_2 = torch.autograd.Variable(torch.randn(batch, input_dim))

    ff_pair = FeedForwardPair.from_params(Params({"input_dim": input_dim, "hidden_dim": output_dim, "activation": "tanh"}))

    r1, r2 = ff_pair(test_input_1, test_input_2)

    assert r1.size() == r2.size() == torch.Size([batch, output_dim])

    test_input_2 = test_input_1.clone()
    r3, r4 = ff_pair(test_input_1, test_input_2)
    assert (r3 == r4).all()