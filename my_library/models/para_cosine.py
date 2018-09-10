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

from my_library.models.feedforward_pair import FeedForwardPair


@Model.register("para_cosine")
class ParaCosine(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 feedforward: FeedForwardPair,
                 margin: float = 1.25,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(ParaCosine, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.feedforward = feedforward
        self.margin = margin

        self.accuracy = BooleanAccuracy()

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

        distance = F.pairwise_distance(fc_p, fc_h)
        prediction = distance < (self.margin / 2.0)
        output_dict = {'distance': distance, "prediction": prediction}

        if label is not None:
            """
            Contrastive loss function.
            Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
            """
            y = label.float()
            l1 = y * torch.pow(distance, 2) / 2.0
            l2 = (1 - y) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2) / 2.0
            loss = torch.mean(l1 + l2)

            self.accuracy(prediction, label.byte())

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
        return {'accuracy': self.accuracy.get_metric(reset)}


