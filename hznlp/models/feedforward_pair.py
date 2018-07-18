"""
A feed-forward neural network for pairs of input.
"""



from typing import Sequence, Union

import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.nn import Activation


class FeedForwardPair(torch.nn.Module):
    """
    This ``Module`` is a feed-forward neural network for pairs of input, a sequence and cross of
    ``Linear`` layers with activation functions in between.
    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of the input.  We assume the input has shape ``(batch_size, input_dim)``.
    num_layers : ``int``
        The number of ``Linear`` layers to apply to the input.
    hidden_dims : ``Union[int, Sequence[int]]``
        The output dimension of each of the ``Linear`` layers.  If this is a single ``int``, we use
        it for all ``Linear`` layers.  If it is a ``Sequence[int]``, ``len(hidden_dims)`` must be
        ``num_layers``.
    activations : ``Union[Callable, Sequence[Callable]]``
        The activation function to use after each ``Linear`` layer.  If this is a single function,
        we use it after all ``Linear`` layers.  If it is a ``Sequence[Callable]``,
        ``len(activations)`` must be ``num_layers``.
    dropout : ``Union[float, Sequence[float]]``, optional
        If given, we will apply this amount of dropout after each layer.  Semantics of ``float``
        versus ``Sequence[float]`` is the same as with other parameters.
    """
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 hidden_dims: Union[int, Sequence[int]],
                 activations: Union[Activation, Sequence[Activation]],
                 dropout: Union[float, Sequence[float]] = 0.0) -> None:

        super(FeedForwardPair, self).__init__()
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers  # type: ignore
        if not isinstance(activations, list):
            activations = [activations] * num_layers  # type: ignore
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers  # type: ignore
        if len(hidden_dims) != num_layers:
            raise ConfigurationError("len(hidden_dims) (%d) != num_layers (%d)" %
                                     (len(hidden_dims), num_layers))
        if len(activations) != num_layers:
            raise ConfigurationError("len(activations) (%d) != num_layers (%d)" %
                                     (len(activations), num_layers))
        if len(dropout) != num_layers:
            raise ConfigurationError("len(dropout) (%d) != num_layers (%d)" %
                                     (len(dropout), num_layers))
        self._activations = activations
        input_dims = [input_dim] + hidden_dims[:-1]
        straight_layers = []
        cross_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            straight_layers.append(torch.nn.Linear(layer_input_dim, layer_output_dim))
            cross_layers.append(torch.nn.Linear(layer_input_dim, layer_output_dim))
        self._straight_layers = torch.nn.ModuleList(straight_layers)
        self._cross_layers = torch.nn.ModuleList(cross_layers)
        dropout_layers = [torch.nn.Dropout(p=value) for value in dropout]
        self._dropout = torch.nn.ModuleList(dropout_layers)
        self._output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self.input_dim

    def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        outputs1, outputs2 = inputs1, inputs2

        for straight, cross, activation, dropout in zip(self._straight_layers, self._cross_layers, self._activations, self._dropout):
            r1 = dropout(activation(straight(outputs1) + cross(outputs2)))
            r2 = dropout(activation(straight(outputs2) + cross(outputs1)))
            outputs1, outputs2 = r1, r2

        return outputs1, outputs2

    # Requires custom logic around the activations (the automatic `from_params`
    # method can't currently instatiate types like `Union[Activation, List[Activation]]`)
    @classmethod
    def from_params(cls, params: Params):
        input_dim = params.pop_int('input_dim')
        num_layers = params.pop_int('num_layers')
        hidden_dims = params.pop('hidden_dims')
        activations = params.pop('activations')
        dropout = params.pop('dropout', 0.0)
        if isinstance(activations, list):
            activations = [Activation.by_name(name)() for name in activations]
        else:
            activations = Activation.by_name(activations)()
        params.assert_empty(cls.__name__)
        return cls(input_dim=input_dim,
                   num_layers=num_layers,
                   hidden_dims=hidden_dims,
                   activations=activations,
                   dropout=dropout)

if __name__ == "__main__":
    from allennlp.common import Params
    torch.manual_seed(999)

    batch = 16
    input_dim = 200
    hidden1 = 100
    hidden2 = 80
    test_input_1 = torch.autograd.Variable(torch.randn(batch, input_dim))
    test_input_2 = torch.autograd.Variable(torch.randn(batch, input_dim))

    ff_pair = FeedForwardPair.from_params(Params({
        "input_dim": input_dim, "num_layers": 2, "hidden_dims": [hidden1, hidden2],
        "activations": ["tanh", "linear"], "dropout": [0.0, 0.0]}))

    r1, r2 = ff_pair(test_input_1, test_input_2)

    assert r1.size() == r2.size() == torch.Size([batch, hidden2])

    test_input_2 = test_input_1.clone()
    r3, r4 = ff_pair(test_input_1, test_input_2)
    assert (r3 == r4).all()
