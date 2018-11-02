from overrides import overrides
import torch

from allennlp.nn.util import masked_softmax, weighted_sum
from allennlp.modules import FeedForward, InputVariationalDropout
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction

torch.set_printoptions(edgeitems=8, linewidth=240)

@Seq2SeqEncoder.register("self_attentive_lstm")
class SelfAttentiveLstm(Seq2SeqEncoder):
    def __init__(self,
                 encoder1: Seq2SeqEncoder,
                 encoder2: Seq2SeqEncoder,
                 similarity_function: SimilarityFunction,
                 projection_feedforward: FeedForward,
                 #inference_encoder: Seq2SeqEncoder,
                 dropout: float = 0.5) -> None:
        super(SelfAttentiveLstm, self).__init__()

        self._encoder1 = encoder1
        self._encoder2 = encoder2
        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._projection_feedforward = projection_feedforward
        #self._inference_encoder = inference_encoder

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

    def get_input_dim(self):
        return self._encoder1.get_input_dim()

    def get_output_dim(self):
        return self._projection_feedforward.get_output_dim()

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
        A tensor of shape (batch_size, timesteps, output_dim),
        """
        # apply dropout for LSTM

        inputs = inputs * mask.unsqueeze(-1).float()

        if self.rnn_input_dropout:
            inputs = self.rnn_input_dropout(inputs)

        # encode inputs in two different ways
        encoded_inputs1 = self._encoder1(inputs, mask)
        encoded_inputs2 = self._encoder2(inputs, mask)

        # Shape: (batch_size, timesteps, timesteps)
        similarity_matrix = self._matrix_attention(encoded_inputs1, encoded_inputs2)

        mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        # Shape: (batch_size, timesteps, timesteps)
        attention1 = masked_softmax(similarity_matrix, mask_2d)
        # Shape: (batch_size, timesteps, embedding_dim)
        attended_inputs1 = weighted_sum(encoded_inputs1, attention1)

        # Shape: (batch_size, timesteps, timesteps)
        attention2 = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), mask_2d)
        # Shape: (batch_size, timesteps, embedding_dim)
        attended_inputs2 = weighted_sum(encoded_inputs2, attention2)

        # the "enhancement" layer
        enhanced = torch.cat(
                [attended_inputs1, attended_inputs2,
                 attended_inputs1 - attended_inputs2,
                 attended_inputs1 * attended_inputs2],
                dim=-1
        )

        # The projection layer down to the model dimension.  Dropout is not applied before
        # projection.
        projected = self._projection_feedforward(enhanced)

        # Run the inference layer
        #if self.rnn_input_dropout:
        #    projected = self.rnn_input_dropout(projected)

        #inferenced = self._inference_encoder(projected, mask)

        return projected