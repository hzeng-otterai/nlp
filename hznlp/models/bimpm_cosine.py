from typing import Dict, Optional

from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import BooleanAccuracy

from hznlp.models.matching_layer import MatchingLayer
from hznlp.models.feedforward_pair import FeedForwardPair


@Model.register("bimpm_cosine")
class BiMPMCosine(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 matcher: MatchingLayer,
                 aggregator: Seq2VecEncoder,
                 feedforward: FeedForwardPair,
                 similarity: SimilarityFunction,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BiMPMCosine, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("label")
        self.encoder = encoder
        self.matcher = matcher
        self.aggregator = aggregator
        self.feedforward = feedforward
        self.similarity = similarity

        self.metrics = {
            "accuracy": BooleanAccuracy()
        }
        self.loss = torch.nn.MSELoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
 
        mask_p = util.get_text_field_mask(premise)
        mask_h = util.get_text_field_mask(hypothesis)

        embedded_p = self.text_field_embedder(premise)
        embedded_p = F.dropout(embedded_p, p=0.1, training=self.training)
        encoded_p = self.encoder(embedded_p, mask_p)
        encoded_p = F.dropout(encoded_p, p=0.1, training=self.training)

        embedded_h = self.text_field_embedder(hypothesis)
        embedded_h = F.dropout(embedded_h, p=0.1, training=self.training)
        encoded_h = self.encoder(embedded_h, mask_h)
        encoded_h = F.dropout(encoded_h, p=0.1, training=self.training)

        mv_p, mv_h = self.matcher(encoded_p, mask_p, encoded_h, mask_h)
        agg_p = self.aggregator(mv_p, mask_p)
        agg_p = F.dropout(agg_p, p=0.1, training=self.training)
        agg_h = self.aggregator(mv_h, mask_h)
        agg_h = F.dropout(agg_h, p=0.1, training=self.training)

        fc_p, fc_h = self.feedforward(agg_p, agg_h)

        similarity = self.similarity(fc_p, fc_h)

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

