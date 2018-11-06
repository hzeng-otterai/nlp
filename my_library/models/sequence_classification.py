from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
from torch import nn
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("sequence_classification")
class SequenceClassification(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 embedding_dropout: float,
                 seq2seq_encoder: Seq2SeqEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SequenceClassification, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self._embedding_dropout = nn.Dropout(embedding_dropout)
        self.num_classes = self.vocab.get_vocab_size("label")
        self.seq2seq_encoder = seq2seq_encoder
        self.self_attentive_pooling_projection = nn.Linear(seq2seq_encoder.get_output_dim(), 1)
        self.classifier_feedforward = classifier_feedforward

        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }

        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
 
        mask_tokens = util.get_text_field_mask(tokens)
        embedded_tokens = self.text_field_embedder(tokens)
        dropped_embedded_tokens = self._embedding_dropout(embedded_tokens)

        encoded_tokens = self.seq2seq_encoder(dropped_embedded_tokens, mask_tokens)

        self_attentive_logits = self.self_attentive_pooling_projection(encoded_tokens).squeeze(2)
        self_weights = util.masked_softmax(self_attentive_logits, mask_tokens)
        encoding_result = util.weighted_sum(encoded_tokens, self_weights)

        logits = self.classifier_feedforward(encoding_result)
        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {'logits': logits, 'class_probabilities': class_probabilities}

        if label is not None:
            loss = self.loss(logits, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["class_probabilities"].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

