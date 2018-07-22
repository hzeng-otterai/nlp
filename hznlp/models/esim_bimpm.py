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
                 matcher_fw: MatchingLayer,
                 matcher_bw: MatchingLayer,
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

        self._matcher_fw = matcher_fw
        self._matcher_bw = matcher_bw

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

        # Using BiMPM to calculate matching vectors
        # Shape: (batch_size, seq_length, num_perspective * num_matching)
        dim = encoded_h.size(-1)
        encoded_p_fw, encoded_p_bw = torch.split(encoded_p, dim // 2, dim=-1)
        encoded_h_fw, encoded_h_bw = torch.split(encoded_h, dim // 2, dim=-1)
        mv_p_fw, mv_h_fw = self._matcher_fw(encoded_p_fw, mask_p, encoded_h_fw, mask_h)
        mv_p_bw, mv_h_bw = self._matcher_bw(encoded_p_bw, mask_p, encoded_h_bw, mask_h)
        mv_p = self.dropout(torch.cat(mv_p_fw + mv_p_bw, dim=2))
        mv_h = self.dropout(torch.cat(mv_h_fw + mv_h_bw, dim=2))

        # the "enhancement" layer
        # Shape: (batch_size, p_length, encoding_direction_num * encoding_hidden_dim * 4 + num_perspective * num_matching)
        enhanced_p = torch.cat(
                [encoded_p, attended_h,
                 encoded_p - attended_h,
                 encoded_p * attended_h,
                 mv_p],
                dim=-1
        )
        # Shape: (batch_size, h_length, encoding_direction_num * encoding_hidden_dim * 4 + num_perspective * num_matching)
        enhanced_h = torch.cat(
                [encoded_h, attended_p,
                 encoded_h - attended_p,
                 encoded_h * attended_p,
                 mv_h],
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
        v_ai = self._inference_encoder(projected_enhanced_p, mask_p)
        v_bi = self._inference_encoder(projected_enhanced_h, mask_h)

        # The pooling layer -- max and avg pooling.
        # Shape: (batch_size, inference_direction_num * inference_hidden_dim)
        v_a_max, _ = replace_masked_values(
                v_ai, mask_p.unsqueeze(-1), -1e7
        ).max(dim=1)
        v_b_max, _ = replace_masked_values(
                v_bi, mask_h.unsqueeze(-1), -1e7
        ).max(dim=1)

        v_a_avg = torch.sum(v_ai * mask_p.unsqueeze(-1), dim=1) / torch.sum(
                mask_p, 1, keepdim=True
        )
        v_b_avg = torch.sum(v_bi * mask_h.unsqueeze(-1), dim=1) / torch.sum(
                mask_h, 1, keepdim=True
        )

        # Now concat
        # Shape: (batch_size, inference_direction_num * inference_hidden_dim * 4)
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