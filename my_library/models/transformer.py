from overrides import overrides
import math
import torch
from torch import nn

from allennlp.modules import Seq2SeqEncoder, LayerNorm
from allennlp.nn.util import masked_softmax, weighted_sum


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.

        Parameters
        ----------
        x : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        sublayer : sublayer to convert input to output of the same shape

        Returns
        -------
        A tensor of shape (batch_size, timesteps, input_dim)
        """
        return x + self.dropout(sublayer(self.norm(x)))


class Attention(nn.Module):
    """
    Compute Scaled Dot Product Attention
    """
    def __init__(self, input_dim_per_head, dropout=0.1):
        super().__init__()
        self.input_dim_per_head = input_dim_per_head
        self.scale = math.sqrt(input_dim_per_head)

        if dropout:
            self.dropout_layer = nn.Dropout(p=dropout)
        else:
            self.dropout_layer = None

    def forward(self, query, key, value, mask=None):
        """
        Parameters
        ----------
        query : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, heads, timesteps, input_dim_per_head)
        key : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, heads, timesteps, input_dim_per_head)
        value : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, heads, timesteps, input_dim_per_head)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        A tuple of (result, attention)
        result: A tensor of shape (batch_size, heads, timesteps, input_dim_per_head)
        attention: A tensor of shape (batch_size, heads, timesteps, timesteps)
        """
        assert self.input_dim_per_head == query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        if mask is not None:
            expanded_mask = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).unsqueeze(1)
            p_attn = masked_softmax(scores, expanded_mask)
        else:
            p_attn = nn.functional.softmax(scores, dim=-1)

        if self.dropout_layer is not None:
            p_attn = self.dropout_layer(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, num_heads, input_dim, dropout=0.1):
        super().__init__()
        assert input_dim % num_heads == 0

        # We assume d_v always equals dim_per_head
        self.input_dim = input_dim
        self.dim_per_head = input_dim // num_heads
        self.num_heads = num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(3)])
        self.output_linear = nn.Linear(input_dim, input_dim)
        self.attention = Attention(self.dim_per_head, dropout=dropout)

    def forward(self, inputs, mask=None):
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """
        query, key, value = inputs, inputs, inputs
        batch_size = query.size(0)
        mask_float = mask.unsqueeze(-1).float()

        # 1) Do all the linear projections in batch from input_dim => num_heads x dim_per_head
        # query, key, value: tensors of shape (batch_size, heads, timesteps, input_dim_per_head)
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # x: tensor of shape (batch_size, heads, timesteps, input_dim_per_head)
        x, _ = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        # x: tensor of shape (batch_size, timesteps, input_dim)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.input_dim)

        # tensor of shape (batch_size, timesteps, input_dim)
        return self.output_linear(x)

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    """

    def __init__(self, input_dim, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(input_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(num_heads=attn_heads, input_dim=hidden)
        self.feed_forward = PositionwiseFeedForward(input_dim=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        """
        Parameters
        ----------
        x : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, 1, timesteps, timesteps).

        Returns
        -------
        A tensor of shape (batch_size, timesteps, input_dim)
        """
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class PositionalEmbedding(nn.Module):

    def __init__(self, input_dim, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, input_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, input_dim, 2) * -(math.log(10000.0) / input_dim)).float().exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

@Seq2SeqEncoder.register("transformer")
class Transformer(Seq2SeqEncoder):
    def __init__(self,
                 hidden_dim: int,
                 num_layers: int = 10,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(Transformer, self).__init__()

        self._hidden_dim = hidden_dim

        self._position_embedding = PositionalEmbedding(input_dim=hidden_dim)

        self._transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout) for _ in range(num_layers)])

    def get_input_dim(self):
        return self._hidden_dim

    def get_output_dim(self):
        return self._hidden_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.LongTensor = None) -> torch.FloatTensor:
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.FloatTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).

        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_projection_dim),
        where output_projection_dim = input_dim by default.
        """

        # (1, timesteps, input_dim)
        pos_embedding = self._position_embedding(inputs)

        # (batch_size, 1, timesteps, timesteps)
        # expanded_mask = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).unsqueeze(1)

        # (batch_size, timesteps, input_dim)
        outputs = (inputs + pos_embedding) * mask.unsqueeze(-1).float()

        for transformer in self._transformer_blocks:
            outputs = transformer.forward(outputs, mask)

        return outputs

