from typing import Dict, Optional, List, Any
from overrides import overrides

import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, InputVariationalDropout
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, last_dim_softmax, weighted_sum, replace_masked_values
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy

from hznlp.models.feedforward_pair import FeedForwardPair


@Model.register("esim_cosine")
class ESIMCosine(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 similarity_function: SimilarityFunction,
                 projection_feedforward: FeedForward,
                 inference_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForwardPair,
                 dropout: float = 0.5,
                 margin: float = 1.25,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder

        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._projection_feedforward = projection_feedforward

        self._inference_encoder = inference_encoder

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

        self._output_feedforward = output_feedforward
        
        self._margin = margin

        self._accuracy = BooleanAccuracy()
        
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:

        # Shape: (batch_size, seq_length, embedding_dim)
        embedded_p = self._text_field_embedder(premise)
        embedded_h = self._text_field_embedder(hypothesis)
        
        mask_p = get_text_field_mask(premise).float()
        mask_h = get_text_field_mask(hypothesis).float()

        # apply dropout for LSTM
        if self.rnn_input_dropout:
            embedded_p = self.rnn_input_dropout(embedded_p)
            embedded_h = self.rnn_input_dropout(embedded_h)

        # encode p and h
        # Shape: (batch_size, seq_length, encoding_direction_num * encoding_hidden_dim)
        encoded_p = self._encoder(embedded_p, mask_p)
        encoded_h = self._encoder(embedded_h, mask_h)

        # Shape: (batch_size, p_length, h_length)
        similarity_matrix = self._matrix_attention(encoded_p, encoded_h)

        # Shape: (batch_size, p_length, h_length)
        p2h_attention = last_dim_softmax(similarity_matrix, mask_h)
        # Shape: (batch_size, p_length, encoding_direction_num * encoding_hidden_dim)
        attended_h = weighted_sum(encoded_h, p2h_attention)

        # Shape: (batch_size, h_length, p_length)
        h2p_attention = last_dim_softmax(similarity_matrix.transpose(1, 2).contiguous(), mask_p)
        # Shape: (batch_size, h_length, encoding_direction_num * encoding_hidden_dim)
        attended_p = weighted_sum(encoded_p, h2p_attention)

        # the "enhancement" layer
        # Shape: (batch_size, p_length, encoding_direction_num * encoding_hidden_dim * 4 + num_perspective * num_matching)
        enhanced_p = torch.cat(
                [encoded_p, attended_h,
                 encoded_p - attended_h,
                 encoded_p * attended_h],
                dim=-1
        )
        # Shape: (batch_size, h_length, encoding_direction_num * encoding_hidden_dim * 4 + num_perspective * num_matching)
        enhanced_h = torch.cat(
                [encoded_h, attended_p,
                 encoded_h - attended_p,
                 encoded_h * attended_p],
                dim=-1
        )

        # The projection layer down to the model dimension.  Dropout is not applied before
        # projection.
        # Shape: (batch_size, seq_length, projection_hidden_dim)
        projected_enhanced_p = self._projection_feedforward(enhanced_p)
        projected_enhanced_h = self._projection_feedforward(enhanced_h)

        # Run the inference layer
        if self.rnn_input_dropout:
            projected_enhanced_p = self.rnn_input_dropout(projected_enhanced_p)
            projected_enhanced_h = self.rnn_input_dropout(projected_enhanced_h)
            
        # Shape: (batch_size, seq_length, inference_direction_num * inference_hidden_dim)
        inferenced_p = self._inference_encoder(projected_enhanced_p, mask_p)
        inferenced_h = self._inference_encoder(projected_enhanced_h, mask_h)

        # The pooling layer -- max and avg pooling.
        # Shape: (batch_size, inference_direction_num * inference_hidden_dim)
        pooled_p_max, _ = replace_masked_values(
                inferenced_p, mask_p.unsqueeze(-1), -1e7
        ).max(dim=1)
        pooled_h_max, _ = replace_masked_values(
                inferenced_h, mask_h.unsqueeze(-1), -1e7
        ).max(dim=1)

        pooled_p_avg = torch.sum(inferenced_p * mask_p.unsqueeze(-1), dim=1) / torch.sum(
                mask_p, 1, keepdim=True
        )
        pooled_h_avg = torch.sum(inferenced_h * mask_h.unsqueeze(-1), dim=1) / torch.sum(
                mask_h, 1, keepdim=True
        )

        # Now concat
        # Shape: (batch_size, inference_direction_num * inference_hidden_dim * 2)
        pooled_p_all = torch.cat([pooled_p_avg, pooled_p_max], dim=1)
        pooled_h_all = torch.cat([pooled_h_avg, pooled_h_max], dim=1)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        if self.dropout:
            pooled_p_all = self.dropout(pooled_p_all)
            pooled_h_all = self.dropout(pooled_h_all)

        # Shape: (batch_size, output_feedforward_hidden_dim)
        output_p, output_h = self._output_feedforward(pooled_p_all, pooled_h_all)

        distance = F.pairwise_distance(output_p, output_h)
        prediction = distance < (self._margin / 2.0)
        output_dict = {'distance': distance, "prediction": prediction}

        if label is not None:
            
            """
            Contrastive loss function.
            Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
            """
            y = label.float()
            l1 = y * torch.pow(distance, 2) / 2.0
            l2 = (1 - y) * torch.pow(torch.clamp(self._margin - distance, min=0.0), 2) / 2.0
            loss = torch.mean(l1 + l2)

            self._accuracy(prediction, label.byte())

            output_dict["loss"] = loss

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
