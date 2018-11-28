# coding=utf-8
"""
PyTorch BERT model.
Slightly modified from https://github.com/codertimo/BERT-pytorch
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

import math

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.data import Vocabulary
from allennlp.common import Params

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-base-multilingual': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class BertLayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, hidden_size, eps=1e-6):
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, vocab_size, hidden_size, dropout=0.1, max_position_embeddings=512, type_vocab_size=3):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, attention_probs_dropout_prob=0.1):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = ACT2FN["gelu"]

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class BertLayer(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = BertSelfAttention(num_attention_heads=attn_heads, hidden_size=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)

        self.input_norm = BertLayerNorm(hidden)
        self.input_dropout = nn.Dropout(dropout)

        self.output_norm = BertLayerNorm(hidden)
        self.output_dropout = nn.Dropout(dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        old_x = x
        x = self.input_norm(x)
        x = self.attention(x, attention_mask=mask)
        x = old_x + self.input_dropout(x)

        old_x = x
        x = self.output_norm(x)
        x = self.feed_forward(x)
        x = old_x + self.output_dropout(x)

        return self.dropout(x)


class BertEncoder(nn.Module):
    def __init__(self, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):

        super().__init__()
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [BertLayer(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, attention_mask=None, output_all_encoded_layers=True):
        all_encoder_layers = []
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, attention_mask)
            all_encoder_layers.append(x)

        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BertEmbeddings(vocab_size=vocab_size, hidden_size=hidden)

        # multi-layers transformer blocks, deep network
        self.encoder = BertEncoder(hidden=hidden, n_layers=n_layers, attn_heads=attn_heads, dropout=dropout)

        self.pooler = BertPooler(hidden_size=hidden)

        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """

        initializer_range = 0.02
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.beta.data.normal_(mean=0.0, std=initializer_range)
            module.gamma.data.normal_(mean=0.0, std=initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, token_type_ids=None, attention_mask=None):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        #mask = attention_mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, token_type_ids)

        encoded_layers = self.encoder(x, attention_mask=extended_attention_mask)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output


@TokenEmbedder.register("bert_embedder3")
class BertEmbedder3(TokenEmbedder):
    def __init__(self,
                 vocab_size: int,
                 hidden_dim: int,
                 num_layers: int = 10,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(BertEmbedder3, self).__init__()

        self._hidden_dim = hidden_dim
        self._bert = BertModel(vocab_size,
                          hidden=hidden_dim, 
                          n_layers=num_layers, 
                          attn_heads=num_heads, 
                          dropout=dropout)

    def get_output_dim(self):
        """
        The last dimension of the output, not the shape.
        """
        return self._hidden_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``, required
            A ``(batch_size, num_timesteps)`` tensor representing the byte-pair encodings
            for the current batch.
        Returns
        -------
        ``[torch.Tensor]``
            An embedding representation of the input sequence
            having shape ``(batch_size, sequence_length, embedding_dim)``
        """
        # pylint: disable=arguments-differ

        mask = (inputs != 0).long()
        sequence_output, pooled_output = self._bert(inputs, attention_mask=mask)
        return sequence_output

    # Custom logic requires custom from_params.
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'Embedding':  # type: ignore
        vocab_size = params.pop_int('vocab_size', None)
        if vocab_size is None:
            vocab_namespace = params.pop("vocab_namespace", "tokens")
            vocab_size = vocab.get_vocab_size(vocab_namespace)

        hidden_dim = params.pop_int('hidden_dim')

        return cls(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim
        )
