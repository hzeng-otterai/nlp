from typing import Dict, Optional

from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import BooleanAccuracy

from hznlp.models.feedforward_pair import FeedForwardPair


@Model.register("para_cosine")
class ParaCosine(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 feedforward: FeedForwardPair,
                 margin: float = 0.4,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(ParaCosine, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.feedforward = feedforward
        self.margin = margin

        self.metrics = {
            "accuracy": BooleanAccuracy()
        }

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

        fc_p, fc_h = self.feedforward(encoded_p, encoded_h)

        cos_sim = F.cosine_similarity(fc_p, fc_h)

        prediction = cos_sim > self.margin
        output_dict = {'similarity': cos_sim, "prediction": prediction}

        if label is not None:
            """
            Cosine contrastive loss function.
            Based on: http://anthology.aclweb.org/W16-1617
            """
            y = label.float()
            l1 = y * torch.pow((1.0 - cos_sim), 2) / 4.0
            l2 = (1 - y) * torch.pow(cos_sim * prediction.float(), 2)

            loss = torch.mean(l1 + l2)
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


