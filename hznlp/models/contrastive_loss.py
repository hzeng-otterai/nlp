"""
Contrastive Loss function
"""

import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin: float = 2.0) -> None:

        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.check_type_forward((x0, x1, y))
        # y == 1 similar, y == 0 dissimilar
        # (Y * D^2 + (1 - Y) * (max(0, m−D))^2) / 2

        # euclidian distance
        distance = F.pairwise_distance(x0, x1)

        l1 = y.float() * torch.pow(distance, 2) / 2.0
        l2 = (1 - y).float() * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2) / 2.0
        loss = torch.mean(l1 + l2)
        return loss


class CosineContrastiveLoss(torch.nn.Module):
    """
    Cosine contrastive loss function.
    Based on: http://anthology.aclweb.org/W16-1617
    Maintain 1 for match, 0 for not match.
    If they match, loss is 1/4(1-cos_sim)^2.
    If they don't, it's cos_sim^2 if cos_sim < margin or 0 otherwise.
    Margin in the paper is ~0.4.
    """

    def __init__(self, margin=0.4):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos_sim = F.cosine_similarity(x0, x1)

        l1 = y.float() * torch.pow((1.0-cos_sim), 2) / 4.0
        l2 = (1 - y).float() * torch.pow(cos_sim * torch.gt(cos_sim, self.margin).float(), 2)
        loss = torch.mean(l1 + l2)
        return loss


if __name__ == "__main__":
    torch.manual_seed(999)

    test_input_1 = torch.FloatTensor([[0.6, 0.5, 1.5], [0.6, 0.5, 1.5], [1.2, 1.3, -1.2], [1.2, 1.3, -1.2]])
    test_input_2 = torch.FloatTensor([[-0.5, -0.6, 0.2], [-0.5, -0.6, 0.2], [1.21, 1.29, -1.2], [1.2, 1.3, -1.2]])
    label = torch.LongTensor([1, 0, 1, 0])

    loss_func = ContrastiveLoss()

    l = loss_func(test_input_1, test_input_2, label)

    print("ConstrativeLoss", l)

    loss_func = CosineContrastiveLoss()

    l = loss_func(test_input_1, test_input_2, label)

    print("CosineConstrativeLoss", l)
