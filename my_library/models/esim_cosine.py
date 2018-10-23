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
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum, replace_masked_values
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy

from my_library.models.feedforward_pair import FeedForwardPair


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
        embedded_premise = self._text_field_embedder(premise)
        embedded_hypothesis = self._text_field_embedder(hypothesis)
        
        mask_premise = get_text_field_mask(premise).float()
        mask_hypothesis = get_text_field_mask(hypothesis).float()

        # apply dropout for LSTM
        if self.rnn_input_dropout:
            embedded_premise = self.rnn_input_dropout(embedded_premise)
            embedded_hypothesis = self.rnn_input_dropout(embedded_hypothesis)

        # encode premise and hypothesis
        # Shape: (batch_size, seq_length, encoding_direction_num * encoding_hidden_dim)
        encoded_premise = self._encoder(embedded_premise, mask_premise)
        encoded_hypothesis = self._encoder(embedded_hypothesis, mask_hypothesis)

        # Shape: (batch_size, p_length, h_length)
        similarity_matrix = self._matrix_attention(encoded_premise, encoded_hypothesis)

        # Shape: (batch_size, p_length, h_length)
        p2h_attention = masked_softmax(similarity_matrix, mask_hypothesis)
        # Shape: (batch_size, p_length, encoding_direction_num * encoding_hidden_dim)
        attended_hypothesis = weighted_sum(encoded_hypothesis, p2h_attention)

        # Shape: (batch_size, h_length, p_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), mask_premise)
        # Shape: (batch_size, h_length, encoding_direction_num * encoding_hidden_dim)
        attended_premise = weighted_sum(encoded_premise, h2p_attention)

        # the "enhancement" layer
        # Shape: (batch_size, p_length, encoding_direction_num * encoding_hidden_dim * 4 + num_perspective * num_matching)
        enhanced_premise = torch.cat(
                [encoded_premise, attended_hypothesis,
                 encoded_premise - attended_hypothesis,
                 encoded_premise * attended_hypothesis],
                dim=-1
        )
        # Shape: (batch_size, h_length, encoding_direction_num * encoding_hidden_dim * 4 + num_perspective * num_matching)
        enhanced_hypothesis = torch.cat(
                [encoded_hypothesis, attended_premise,
                 encoded_hypothesis - attended_premise,
                 encoded_hypothesis * attended_premise],
                dim=-1
        )

        # The projection layer down to the model dimension.  Dropout is not applied before
        # projection.
        # Shape: (batch_size, seq_length, projection_hidden_dim)
        projected_enhanced_premise = self._projection_feedforward(enhanced_premise)
        projected_enhanced_hypothesis = self._projection_feedforward(enhanced_hypothesis)

        # Run the inference layer
        if self.rnn_input_dropout:
            projected_enhanced_premise = self.rnn_input_dropout(projected_enhanced_premise)
            projected_enhanced_hypothesis = self.rnn_input_dropout(projected_enhanced_hypothesis)
            
        # Shape: (batch_size, seq_length, inference_direction_num * inference_hidden_dim)
        inferenced_premise = self._inference_encoder(projected_enhanced_premise, mask_premise)
        inferenced_hypothesis = self._inference_encoder(projected_enhanced_hypothesis, mask_hypothesis)

        # The pooling layer -- max and avg pooling.
        # Shape: (batch_size, inference_direction_num * inference_hidden_dim)
        pooled_premise_max, _ = replace_masked_values(
                inferenced_premise, mask_premise.unsqueeze(-1), -1e7
        ).max(dim=1)
        pooled_hypothesis_max, _ = replace_masked_values(
                inferenced_hypothesis, mask_hypothesis.unsqueeze(-1), -1e7
        ).max(dim=1)

        pooled_premise_avg = torch.sum(inferenced_premise * mask_premise.unsqueeze(-1), dim=1) / torch.sum(
                mask_premise, 1, keepdim=True
        )
        pooled_hypothesis_avg = torch.sum(inferenced_hypothesis * mask_hypothesis.unsqueeze(-1), dim=1) / torch.sum(
                mask_hypothesis, 1, keepdim=True
        )

        # Now concat
        # Shape: (batch_size, inference_direction_num * inference_hidden_dim * 2)
        pooled_premise_all = torch.cat([pooled_premise_avg, pooled_premise_max], dim=1)
        pooled_hypothesis_all = torch.cat([pooled_hypothesis_avg, pooled_hypothesis_max], dim=1)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        if self.dropout:
            pooled_premise_all = self.dropout(pooled_premise_all)
            pooled_hypothesis_all = self.dropout(pooled_hypothesis_all)

        # Shape: (batch_size, output_feedforward_hidden_dim)
        output_premise, output_hypothesis = self._output_feedforward(pooled_premise_all, pooled_hypothesis_all)

        distance = F.pairwise_distance(output_premise, output_hypothesis)
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
