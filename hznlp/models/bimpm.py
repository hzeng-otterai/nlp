from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

from hznlp.models.matching_layer import MatchingLayer


@Model.register("bimpm")
class BiMPM(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 matcher: MatchingLayer,
                 aggregator: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BiMPM, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("label")
        self.encoder = encoder
        self.matcher = matcher
        self.aggregator = aggregator
        self.classifier_feedforward = classifier_feedforward

        self.accuracy_metric = CategoricalAccuracy()
        self.f1_metric = F1Measure(positive_label=1)

        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                s1: Dict[str, torch.LongTensor],
                s2: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
 
        mask_s1 = util.get_text_field_mask(s1)
        mask_s2 = util.get_text_field_mask(s2)

        embedded_s1 = self.text_field_embedder(s1)
        encoded_s1 = self.encoder(embedded_s1, mask_s1)

        embedded_s2 = self.text_field_embedder(s2)
        encoded_s2 = self.encoder(embedded_s2, mask_s2)

        mv_s1, mv_s2 = self.matcher(encoded_s1, encoded_s2)
        agg_s1 = self.aggregator(mv_s1, mask_s1)
        agg_s2 = self.aggregator(mv_s2, mask_s2)

        logits = self.classifier_feedforward(torch.cat([agg_s1, agg_s2], dim=-1))

        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label.squeeze(-1))
            self.accuracy_metric(logits, label.squeeze(-1))
            self.f1_metric(logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy = self.accuracy_metric.get_metric(reset)
        precision, recall, f1 = self.f1_metric.get_metric(reset)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

