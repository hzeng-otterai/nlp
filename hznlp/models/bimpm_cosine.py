from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.nn import Activation
from allennlp.training.metrics import BooleanAccuracy

from hznlp.models.matching_layer import MatchingLayer


@Model.register("bimpm_cosine")
class BiMPMCosine(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 matcher: MatchingLayer,
                 aggregator: Seq2VecEncoder,
                 feedforward_straight: FeedForward,
                 feedforward_cross: FeedForward,
                 feedforward_activation: Activation,
                 similarity: SimilarityFunction,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BiMPMCosine, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("label")
        self.encoder = encoder
        self.matcher = matcher
        self.aggregator = aggregator
        self.feedforward_straight = feedforward_straight
        self.feedforward_cross = feedforward_cross
        self.feedforward_activation = feedforward_activation
        self.similarity = similarity

        self.metrics = {
            "accuracy": BooleanAccuracy()
        }
        self.loss = torch.nn.MSELoss()

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

        fc_s1 = self.feedforward_activation(self.feedforward_straight(agg_s1) + self.feedforward_cross(agg_s2))
        fc_s2 = self.feedforward_activation(self.feedforward_straight(agg_s2) + self.feedforward_cross(agg_s1))

        similarity = self.similarity(fc_s1, fc_s2)

        prob = (similarity + 1) * 0.5
        prob = torch.clamp(prob, min=0.0, max=1.0)
        prediction = prob > 0.5
        output_dict = {'prob': prob, "prediction": prediction}

        if label is not None:
            loss = self.loss(prob, label.squeeze(-1).float())
            for metric in self.metrics.values():
                metric(prediction, label.squeeze(-1).byte())
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        prob = output_dict["prob"]
        predictions = prob > 0.5

        predictions = predictions.cpu().data.numpy()
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in predictions]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BiMPMCosine':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        matcher = MatchingLayer.from_params(params.pop("matcher", {}))
        aggregator = Seq2VecEncoder.from_params(params.pop("aggregator"))
        feedforward_straight = FeedForward.from_params(params.pop("feedforward_straight"))
        feedforward_cross = FeedForward.from_params(params.pop("feedforward_cross"))
        feedforward_activation = Activation.by_name(params.pop("feedforward_activation", "linear"))()
        similarity = SimilarityFunction.from_params(params.pop("similarity"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder,
                   matcher=matcher,
                   aggregator=aggregator,
                   feedforward_straight=feedforward_straight,
                   feedforward_cross=feedforward_cross,
                   feedforward_activation=feedforward_activation,
                   similarity = similarity,
                   initializer=initializer,
                   regularizer=regularizer)
