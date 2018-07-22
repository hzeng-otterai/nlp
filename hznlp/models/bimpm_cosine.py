from typing import Dict, Optional

from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
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
                 matcher_fw: MatchingLayer,
                 matcher_bw: MatchingLayer,
                 aggregator: Seq2VecEncoder,
                 feedforward: FeedForwardPair,
                 margin: float = 0.4,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BiMPMCosine, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.matcher_fw = matcher_fw
        self.matcher_bw = matcher_bw
        self.aggregator = aggregator
        self.feedforward = feedforward
        self.margin = margin

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
        encoded_p = self.encoder(embedded_p, mask_p)

        embedded_h = self.text_field_embedder(hypothesis)
        encoded_h = self.encoder(embedded_h, mask_h)

        dim = encoded_h.size(-1)
        encoded_p_fw, encoded_p_bw = torch.split(encoded_p, dim // 2, dim=-1)
        encoded_h_fw, encoded_h_bw = torch.split(encoded_h, dim // 2, dim=-1)

        mv_p_fw, mv_h_fw = self.matcher_fw(encoded_p_fw, mask_p, encoded_h_fw, mask_h)
        mv_p_bw, mv_h_bw = self.matcher_bw(encoded_p_bw, mask_p, encoded_h_bw, mask_h)

        mv_p = torch.cat(mv_p_fw + mv_p_bw, dim=2)
        mv_h = torch.cat(mv_h_fw + mv_h_bw, dim=2)

        agg_p = self.aggregator(mv_p, mask_p)
        agg_h = self.aggregator(mv_h, mask_h)

        fc_p, fc_h = self.feedforward(agg_p, agg_h)

        cos_sim = F.cosine_similarity(fc_p, fc_h)

        prediction = cos_sim > self.margin
        output_dict = {'similarity': cos_sim, "prediction": prediction}

        if label is not None:
            new_label = label * 2 - 1
            loss = self.loss(cos_sim, new_label.float())

            for metric in self.metrics.values():
                metric(prediction, label.byte())
                
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        predictions = output_dict["prediction"]

        predictions = predictions.cpu().data.numpy()
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in predictions]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

