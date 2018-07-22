import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common.registrable import FromParams
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, replace_masked_values



def mpm(v1, v2, w):
    """
    Calculate multi-perspective cosine matching between time-steps of vectors
    of the same length.

    :param v1: (batch, seq_len, hidden_size)
    :param v2: (batch, seq_len, hidden_size)
    :param w: (num_perspective, hidden_size)
    :return: (batch, seq_len, num_perspective)
    """
    batch_size, seq_len, hidden_size = v1.shape
    num_perspective = w.size(0)

    assert v1.shape == v2.shape and w.size(1) == hidden_size

    # (batch, seq_len, 1)
    mv_1 = F.cosine_similarity(v1, v2, 2).unsqueeze(2)

    # (1, 1, num_perspective, hidden_size)
    w = w.unsqueeze(0).unsqueeze(0)

    # (batch, seq_len, num_perspective, hidden_size)
    v1 = w * v1.unsqueeze(2).expand(-1, -1, num_perspective, -1)
    v2 = w * v2.unsqueeze(2).expand(-1, -1, num_perspective, -1)

    mv_many = F.cosine_similarity(v1, v2, dim=3)

    return mv_1, mv_many


def mpm_pairwise(v1, v2, w):
    """
    Calculate multi-perspective cosine matching between each time step of
    one vector and each time step of another vector.

    :param v1: (batch, seq_len1, hidden_size)
    :param v2: (batch, seq_len2, hidden_size)
    :param w: (num_perspective, hidden_size)
    :return: (batch, seq_len1, seq_len2, num_perspective)
    """

    num_perspective = w.size(0)

    # (1, num_perspective, 1, hidden_size)
    w = w.unsqueeze(0).unsqueeze(2)

    # (batch, num_perspective, seq_len, hidden_size)
    v1 = w * v1.unsqueeze(1).expand(-1, num_perspective, -1, -1)
    v2 = w * v2.unsqueeze(1).expand(-1, num_perspective, -1, -1)

    # (batch, num_perspective, seq_len, 1)
    v1_norm = v1.norm(p=2, dim=3, keepdim=True)
    v2_norm = v2.norm(p=2, dim=3, keepdim=True)

    # (batch, num_perspective, seq_len1, seq_len2)
    n = torch.matmul(v1, v2.transpose(2, 3))
    d = v1_norm * v2_norm.transpose(2, 3)

    # (batch, seq_len1, seq_len2, num_perspective)
    m = div_safe(n, d).permute(0, 2, 3, 1)

    return m


def cosine_pairwise(v1, v2):
    """
    Calculate cosine similarity between each time step of
    one vector and each time step of another vector.

    :param v1: (batch, seq_len1, hidden_size)
    :param v2: (batch, seq_len2, hidden_size)
    :return: (batch, seq_len1, seq_len2)
    """

    batch_size = v1.size(0)
    assert batch_size == v2.size(0) and v1.size(2) == v2.size(2)

    # (batch, seq_len1, 1)
    v1_norm = v1.norm(p=2, dim=2, keepdim=True)
    # (batch, 1, seq_len2)
    v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

    # (batch, seq_len1, seq_len2)
    a = torch.bmm(v1, v2.permute(0, 2, 1))
    d = v1_norm * v2_norm

    result = div_safe(a, d)

    return result


def div_safe(n, d, eps=1e-8):
    # too small values are replaced by 1e-8 to prevent it from exploding.
    return n / torch.clamp(d, min=eps)


def masked_max(vector, mask, dim, keepdim=False, min_val=-1e7):
    """
    :param vector: vector to calculate max
    :param mask: mask of the vector, vector and mask must be broadcastable
    :param dim: the dimension to calculate max
    """
    value, _ = replace_masked_values(vector, mask, min_val).max(dim=dim, keepdim=keepdim)
    m, _ = mask.expand(vector.shape).max(dim=dim, keepdim=keepdim)
    value = value * m
    return value


def masked_mean(vector, mask, dim, keepdim=False):
    """
    :param vector: vector to calculate max
    :param mask: mask of the vector, vector and mask must be broadcastable
    :param dim: the dimension to calculate mean
    """
    value_sum = torch.sum(vector * mask, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask.expand(vector.shape), dim=dim, keepdim=keepdim)
    return div_safe(value_sum, value_count.float())


class MatchingLayer(nn.Module, FromParams):
    def __init__(self,
                 is_forward: bool,
                 hidden_dim: int = 100,
                 num_perspective: int = 20,
                 wo_full_match: bool = False,
                 wo_maxpool_match: bool = False,
                 wo_attentive_match: bool = False,
                 wo_max_attentive_match: bool = False) -> None:
        super(MatchingLayer, self).__init__()

        self.is_forward = is_forward
        self.hidden_dim = hidden_dim
        self.num_perspective = num_perspective
        self.wo_full_match = wo_full_match
        self.wo_maxpool_match = wo_maxpool_match
        self.wo_attentive_match = wo_attentive_match
        self.wo_max_attentive_match = wo_max_attentive_match
        self.num_matching = 4 - (
                    int(wo_full_match) + int(wo_maxpool_match) + int(wo_attentive_match) + int(wo_max_attentive_match))
        assert self.num_matching > 0

        self.params = nn.ParameterList(
            [nn.Parameter(torch.rand(num_perspective, hidden_dim)) for i in range(self.num_matching)])

        for para in self.params:
            nn.init.kaiming_normal_(para)

    def forward(self, context_p, mask_p, context_h, mask_h):
        """
        Given the representations of two sentences from BiLSTM, apply four bilateral
        matching functions between premise and hypothesis in one direction

        Parameters
        ----------
        :param context_p: Tensor of shape (batch_size, seq_len, context_rnn_hidden_dim)
            representing premise as encoded by the forward and backward layer of a BiLSTM.
        :param mask_p: Binary Tensor of shape (batch_size, seq_len), indicating which
            positions in premise are padding (0) and which are not (1).
        :param context_h: Tensor of shape (batch_size, seq_len, context_rnn_hidden_dim)
            representing hypothesis as encoded by the forward and backward layer of a BiLSTM.
        :param mask_h: Binary Tensor of shape (batch_size, seq_len), indicating which
            positions in hypothesis are padding (0) and which are not (1).
        :return (mv_p, mv_h): Matching vectors for premise and hypothesis, each of shape
            (batch, seq_len, num_perspective * num_matching)
        """

        # (batch, seq_len, hidden_dim)
        assert context_p.size(-1) == context_h.size(-1) == self.hidden_dim
        seq_len_p, seq_len_h = context_p.size(1), context_h.size(1)

        # (batch,)
        len_p = get_lengths_from_binary_sequence_mask(mask_p)
        len_h = get_lengths_from_binary_sequence_mask(mask_h)

        # (batch, 1, hidden_dim)
        last_token_p = torch.clamp(len_p - 1, min=0).view(-1, 1, 1).expand(-1, 1, self.hidden_dim)
        last_token_h = torch.clamp(len_h - 1, min=0).view(-1, 1, 1).expand(-1, 1, self.hidden_dim)

        # make mask broadcastable with other vectors
        # (batch, seq_len, 1)
        mask_p = mask_p.float().unsqueeze(-1)
        mask_h = mask_h.float().unsqueeze(-1)

        # explicitly set masked weights to zero
        context_p = context_p * mask_p
        context_h = context_h * mask_h

        # array to keep the matching vectors for premise and hypothesis
        mv_p, mv_h = [], []

        # First calculate the cosine similarities between each forward
        # (or backward) contextual embedding and every forward (or backward)
        # contextual embedding of the other sentence.

        # (batch, seq_len1, seq_len2)
        cosine_sim = cosine_pairwise(context_p, context_h)

        # (batch, seq_len1, 1)
        cosine_max_p = masked_max(cosine_sim, mask_p, dim=2, keepdim=True)
        cosine_mean_p = masked_mean(cosine_sim, mask_p, dim=2, keepdim=True)

        # (batch, seq_len2, 1)
        cosine_max_h = masked_max(cosine_sim.permute(0, 2, 1), mask_h, dim=2, keepdim=True)
        cosine_mean_h = masked_mean(cosine_sim.permute(0, 2, 1), mask_h, dim=2, keepdim=True)

        mv_p.extend([cosine_max_p, cosine_mean_p])
        mv_h.extend([cosine_max_h, cosine_mean_h])

        mv_idx = 0

        # 1. Full-Matching
        # Each time step of forward (or backward) contextual embedding of one sentence
        # is compared with the last time step of the forward (or backward)
        # contextual embedding of the other sentence
        if not self.wo_full_match:

            # (batch, 1, hidden_size)
            if self.is_forward:
                context_p_last = context_p.gather(1, last_token_p)
                context_h_last = context_h.gather(1, last_token_h)
            else:
                context_p_last = context_p[:, 0:1, :]
                context_h_last = context_h[:, 0:1, :]

            # (batch, seq_len, num_perspective)
            mv_p_full = mpm(context_p, context_h_last.expand(-1, seq_len_p, -1), self.params[mv_idx])
            mv_h_full = mpm(context_h, context_p_last.expand(-1, seq_len_h, -1), self.params[mv_idx])

            mv_p.extend(mv_p_full)
            mv_h.extend(mv_h_full)

            mv_idx += 1

        # 2. Maxpooling-Matching
        # Each time step of forward (or backward) contextual embedding of one sentence
        # is compared with every time step of the forward (or backward)
        # contextual embedding of the other sentence, and only the max value of each
        # dimension is retained.
        if not self.wo_maxpool_match:
            # (batch, seq_len1, seq_len2, num_perspective)
            mv_max = mpm_pairwise(context_p, context_h, self.params[mv_idx])

            # (batch, seq_len1, num_perspective)
            mv_p_max = masked_max(mv_max, mask_p.unsqueeze(-1), dim=2)
            mv_p_mean = masked_mean(mv_max, mask_p.unsqueeze(-1), dim=2)

            # (batch, seq_len2, seq_len1, num_perspective)
            mv_max = mv_max.permute(0, 2, 1, 3)

            # (batch, seq_len2, num_perspective)
            mv_h_max = masked_max(mv_max, mask_h.unsqueeze(-1), dim=2)
            mv_h_mean = masked_mean(mv_max, mask_h.unsqueeze(-1), dim=2)

            mv_p.extend([mv_p_max, mv_p_mean])
            mv_h.extend([mv_h_max, mv_h_mean])

            mv_idx += 1

        # 3. Attentive-Matching
        # Each forward (or backward) similarity is taken as the weight
        # of the forward (or backward) contextual embedding, and calculate an
        # attentive vector for the sentence by weighted summing all its
        # contextual embeddings.
        # Finally match each forward (or backward) contextual embedding
        # with its corresponding attentive vector.

        # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_h = context_h.unsqueeze(1) * cosine_sim.unsqueeze(3)

        # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_p = context_p.unsqueeze(2) * cosine_sim.unsqueeze(3)

        if not self.wo_attentive_match:
            # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) ->
            # (batch, seq_len1, hidden_size)
            att_mean_h = div_safe(att_h.sum(dim=2), cosine_sim.sum(dim=2, keepdim=True))

            # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) ->
            # (batch, seq_len2, hidden_size)
            att_mean_p = div_safe(att_p.sum(dim=1), cosine_sim.sum(dim=1, keepdim=True).permute(0, 2, 1))

            # (batch, seq_len, num_perspective)
            mv_p_att_mean = mpm(context_p, att_mean_h, self.params[mv_idx])
            mv_h_att_mean = mpm(context_h, att_mean_p, self.params[mv_idx])
            mv_p.extend(mv_p_att_mean)
            mv_h.extend(mv_h_att_mean)

            mv_idx += 1

        # 4. Max-Attentive-Matching
        # Pick the contextual embeddings with the highest cosine similarity as the attentive
        # vector, and match each forward (or backward) contextual embedding with its
        # corresponding attentive vector.
        if not self.wo_max_attentive_match:
            # (batch, seq_len, hidden_size)
            att_max_h = masked_max(att_h, mask_p.unsqueeze(-1), dim=2)
            att_max_p = masked_max(att_p.permute(0, 2, 1, 3), mask_h.unsqueeze(-1), dim=2)

            # (batch, seq_len, num_perspective)
            mv_p_att_max = mpm(context_p, att_max_h, self.params[mv_idx])
            mv_h_att_max = mpm(context_h, att_max_p, self.params[mv_idx])

            mv_p.extend(mv_p_att_max)
            mv_h.extend(mv_h_att_max)

            mv_idx += 1

        return mv_p, mv_h


if __name__ == "__main__":

    from allennlp.common import Params

    torch.set_printoptions(linewidth=200, edgeitems=8)
    torch.manual_seed(999)

    batch = 16
    len1, len2 = 21, 24
    seq_len1 = torch.randint(low=len1 - 10, high=len1 + 1, size=(batch,)).long()
    seq_len2 = torch.randint(low=len2 - 10, high=len2 + 1, size=(batch,)).long()

    mask1 = []
    for l in seq_len1:
        mask1.append([1] * l.item() + [0] * (len1 - l.item()))
    mask1 = torch.FloatTensor(mask1)
    mask2 = []
    for l in seq_len2:
        mask2.append([1] * l.item() + [0] * (len2 - l.item()))
    mask2 = torch.FloatTensor(mask2)

    dim = 200
    num_perspective = 20
    test1 = torch.randn(batch, len1, dim)
    test2 = torch.randn(batch, len2, dim)
    test1 = test1 * mask1.view(-1, len1, 1).expand(-1, len1, dim)
    test2 = test2 * mask2.view(-1, len2, 1).expand(-1, len2, dim)

    test1_fw, test1_bw = torch.split(test1, dim // 2, dim=-1)
    test2_fw, test2_bw = torch.split(test2, dim // 2, dim=-1)

    ml_fw = MatchingLayer.from_params(Params({"is_forward": True, "num_perspective": num_perspective}))
    ml_bw = MatchingLayer.from_params(Params({"is_forward": False, "num_perspective": num_perspective}))

    vecs_p_fw, vecs_h_fw = ml_fw(test1_fw, mask1, test2_fw, mask2)
    vecs_p_bw, vecs_h_bw = ml_bw(test1_bw, mask1, test2_bw, mask2)
    vecs_p, vecs_h = torch.cat(vecs_p_fw + vecs_p_bw, dim=2), torch.cat(vecs_h_fw + vecs_h_bw, dim=2)

    assert vecs_p.size() == torch.Size([batch, len1, 4 + 6 * (num_perspective + 1) + 4 * num_perspective])
    assert vecs_h.size() == torch.Size([batch, len2, 4 + 6 * (num_perspective + 1) + 4 * num_perspective])

    result_len_p = get_lengths_from_binary_sequence_mask(vecs_p > 0.0)
    result_len_h = get_lengths_from_binary_sequence_mask(vecs_h > 0.0)

    print("vecs_p", vecs_p.shape, vecs_p)
    print("vecs_h", vecs_h.shape, vecs_h)