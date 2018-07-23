"""
Loss functions for Siamese networks
"""

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class ContrastiveLoss(_Loss):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin: float = 1.25, size_average: bool = True, reduce: bool = True) -> None:

        super(ContrastiveLoss, self).__init__(size_average, reduce)
        self.margin = margin

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param x0: first input (batch_size, dimensions)
        :param x1: second input (batch_size, dimensions)
        :param y: target (batch_size,)
            y == 1 similar, y == 0 dissimilar
        :return: Y * 1/2 * distance^2 + (1 - Y) * 1/2 * max(0, margin−distance)^2
        """

        # euclidian distance
        distance = F.pairwise_distance(x0, x1)

        l1 = y.float() * torch.pow(distance, 2) / 2.0
        l2 = (1 - y).float() * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2) / 2.0
        loss = l1 + l2

        if not self.reduce:
            return loss
        elif self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class CosineContrastiveLoss(_Loss):
    """
    Cosine contrastive loss function.
    Based on: http://anthology.aclweb.org/W16-1617
    If they match, loss+ is 1/4(1-cos_sim)^2.
    If they don't, loss- is cos_sim^2 if cos_sim < margin or 0 otherwise.
    Margin in the paper is ~0.4.
    (Here I modify loss- to max(cos_sim, 0)^2 to make it continous,
    and the margin is modified to 0.13653 where loss+ == loss-)
    """

    def __init__(self, margin: float = 0.13653, size_average: bool = True, reduce: bool = True) -> None:
        super(CosineContrastiveLoss, self).__init__(size_average, reduce)
        self.margin = margin

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param x0: first input (batch_size, dimensions)
        :param x1: second input (batch_size, dimensions)
        :param y: target (batch_size,)
            y == 1 similar, y == 0 dissimilar
        :return: Y * 1/4(1-cos_sim)^2 + (1 - Y) * max(0, cos_sim)^2
        """
        cos_sim = F.cosine_similarity(x0, x1)

        l1 = y.float() * torch.pow((1.0 - cos_sim), 2) / 4.0
        l2 = (1 - y).float() * torch.pow(cos_sim.clamp(min=0), 2)
        loss = l1 + l2

        if not self.reduce:
            return loss
        elif self.size_average:
            return loss.mean()
        else:
            return loss.sum()


if __name__ == "__main__":

    test_input_1 = torch.FloatTensor([[0.6, 0.5, 1.5], [0.6, 0.5, 1.5], [1.2, 1.3, -1.2], [1.2, 1.3, -1.2]])
    test_input_2 = torch.FloatTensor([[-0.5, -0.6, 0.2], [-0.5, -0.6, 0.2], [1.21, 1.29, -1.2], [1.2, 1.3, -1.2]])
    label = torch.LongTensor([1, 0, 1, 0])

    loss_func = ContrastiveLoss(reduce=False)

    l = loss_func(test_input_1, test_input_2, label)

    print("ConstrativeLoss", l)

    loss_func = CosineContrastiveLoss(reduce=False)

    l = loss_func(test_input_1, test_input_2, label)

    print("CosineConstrativeLoss", l)
