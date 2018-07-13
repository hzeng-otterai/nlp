from typing import Dict, Optional, List, Any

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, InputVariationalDropout
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, last_dim_softmax, weighted_sum, replace_masked_values
from allennlp.training.metrics import CategoricalAccuracy

from hznlp.models.matching_layer import MatchingLayer


@Model.register("esim_bimpm")
class ESIMBiMPM(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 similarity_function: SimilarityFunction,
                 projection_feedforward: FeedForward,
                 matcher: MatchingLayer,
                 inference_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 output_logit: FeedForward,
                 dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder

        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._projection_feedforward = projection_feedforward

        self._matcher = matcher

        self._inference_encoder = inference_encoder

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

        self._output_feedforward = output_feedforward
        self._output_logit = output_logit

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                s1: Dict[str, torch.LongTensor],
                s2: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
               ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        s1 : Dict[str, torch.LongTensor]
            From a ``TextField``
        s2 : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata containing the original tokenization of the s1 and
            s2 with 's1_tokens' and 's2_tokens' keys respectively.

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        # Shape: (batch_size, seq_length, embedding_dim)
        embedded_s1 = self._text_field_embedder(s1)
        embedded_s2 = self._text_field_embedder(s2)
        s1_mask = get_text_field_mask(s1).float()
        s2_mask = get_text_field_mask(s2).float()

        # apply dropout for LSTM
        if self.rnn_input_dropout:
            embedded_s1 = self.rnn_input_dropout(embedded_s1)
            embedded_s2 = self.rnn_input_dropout(embedded_s2)

        # encode s1 and s2
        # Shape: (batch_size, seq_length, encoding_layer_num * encoding_hidden_dim)
        encoded_s1 = self._encoder(embedded_s1, s1_mask)
        encoded_s2 = self._encoder(embedded_s2, s2_mask)

        # Shape: (batch_size, s1_length, s2_length)
        similarity_matrix = self._matrix_attention(encoded_s1, encoded_s2)

        # Shape: (batch_size, s1_length, s2_length)
        p2h_attention = last_dim_softmax(similarity_matrix, s2_mask)
        # Shape: (batch_size, s1_length, encoding_layer_num * encoding_hidden_dim)
        attended_s2 = weighted_sum(encoded_s2, p2h_attention)

        # Shape: (batch_size, s2_length, s1_length)
        h2p_attention = last_dim_softmax(similarity_matrix.transpose(1, 2).contiguous(), s1_mask)
        # Shape: (batch_size, s2_length, encoding_layer_num * encoding_hidden_dim)
        attended_s1 = weighted_sum(encoded_s1, h2p_attention)

        # Using BiMPM to calculate matching vectors
        # Shape: (batch_size, seq_length, num_perspective * num_matching)
        mv_s1, mv_s2 = self._matcher(encoded_s1, encoded_s2)

        # the "enhancement" layer
        # Shape: (batch_size, s1_length, encoding_layer_num * encoding_hidden_dim * 4 + num_perspective * num_matching)
        s1_enhanced = torch.cat(
                [encoded_s1, attended_s2,
                 encoded_s1 - attended_s2,
                 encoded_s1 * attended_s2,
                 mv_s1],
                dim=-1
        )
        # Shape: (batch_size, s2_length, encoding_layer_num * encoding_hidden_dim * 4 + num_perspective * num_matching)
        s2_enhanced = torch.cat(
                [encoded_s2, attended_s1,
                 encoded_s2 - attended_s1,
                 encoded_s2 * attended_s1,
                 mv_s2],
                dim=-1
        )

        # The projection layer down to the model dimension.  Dropout is not applied before
        # projection.
        # Shape: (batch_size, seq_length, projection_hidden_dim)
        projected_enhanced_s1 = self._projection_feedforward(s1_enhanced)
        projected_enhanced_s2 = self._projection_feedforward(s2_enhanced)

        # Run the inference layer
        if self.rnn_input_dropout:
            projected_enhanced_s1 = self.rnn_input_dropout(projected_enhanced_s1)
            projected_enhanced_s2 = self.rnn_input_dropout(projected_enhanced_s2)
        # Shape: (batch_size, seq_length, inference_layer_num * inference_hidden_dim)
        v_ai = self._inference_encoder(projected_enhanced_s1, s1_mask)
        v_bi = self._inference_encoder(projected_enhanced_s2, s2_mask)

        # The pooling layer -- max and avg pooling.
        # Shape: (batch_size, inference_layer_num * inference_hidden_dim)
        v_a_max, _ = replace_masked_values(
                v_ai, s1_mask.unsqueeze(-1), -1e7
        ).max(dim=1)
        v_b_max, _ = replace_masked_values(
                v_bi, s2_mask.unsqueeze(-1), -1e7
        ).max(dim=1)

        v_a_avg = torch.sum(v_ai * s1_mask.unsqueeze(-1), dim=1) / torch.sum(
                s1_mask, 1, keepdim=True
        )
        v_b_avg = torch.sum(v_bi * s2_mask.unsqueeze(-1), dim=1) / torch.sum(
                s2_mask, 1, keepdim=True
        )

        # Now concat
        # Shape: (batch_size, inference_layer_num * inference_hidden_dim * 4)
        v_all = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        if self.dropout:
            v_all = self.dropout(v_all)

        # Shape: (batch_size, output_feedforward_hidden_dim)
        output_hidden = self._output_feedforward(v_all)
        # Shape: (batch_size, output_logit_hidden_dim)
        label_logits = self._output_logit(output_hidden)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}