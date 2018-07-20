import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.common.registrable import FromParams
from allennlp.nn.util import get_lengths_from_binary_sequence_mask


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


class MatchingLayer(nn.Module, FromParams):
    def __init__(self,
                 hidden_dim: int = 100,
                 num_perspective: int = 20,
                 wo_full_match: bool = False,
                 wo_maxpool_match: bool = False,
                 wo_attentive_match: bool = False,
                 wo_max_attentive_match: bool = False) -> None:
        super(MatchingLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_perspective = num_perspective
        self.wo_full_match = wo_full_match
        self.wo_maxpool_match = wo_maxpool_match
        self.wo_attentive_match = wo_attentive_match
        self.wo_max_attentive_match = wo_max_attentive_match
        self.num_matching = 8 - 2 * (
                    int(wo_full_match) + int(wo_maxpool_match) + int(wo_attentive_match) + int(wo_max_attentive_match))
        assert self.num_matching > 0

        self.params = nn.ParameterList(
            [nn.Parameter(torch.rand(num_perspective, hidden_dim)) for i in range(self.num_matching)])

        for para in self.params:
            nn.init.kaiming_normal_(para)

    def forward(self, context_p, mask_p, context_h, mask_h):
        """
        Given the representations of two sentences from BiLSTM, apply four bilateral
        matching functions between premise and hypothesis in both directions

        Parameters
        ----------
        :param context_p: Tensor of shape (batch_size, seq_len, context_rnn_hidden dim * 2)
            representing premise as encoded by the forward and backward layer of a BiLSTM.
        :param mask_p: Binary Tensor of shape (batch_size, seq_len), indicating which
            positions in premise are padding (0) and which are not (1).
        :param context_h: Tensor of shape (batch_size, seq_len, context_rnn_hidden dim * 2)
            representing hypothesis as encoded by the forward and backward layer of a BiLSTM.
        :param mask_h: Binary Tensor of shape (batch_size, seq_len), indicating which
            positions in hypothesis are padding (0) and which are not (1).
        :return (mv_p, mv_h): Matching vectors for premise and hypothesis, each of shape
            (batch, seq_len, num_perspective * num_matching)
        """

        # (batch, seq_len, hidden_dim)
        assert context_p.size(-1) == context_h.size(-1) == self.hidden_dim * 2
        seq_len_p, seq_len_h = context_p.size(1), context_h.size(1)

        context_p_fw, context_p_bw = torch.split(context_p, self.hidden_dim, dim=-1)
        context_h_fw, context_h_bw = torch.split(context_h, self.hidden_dim, dim=-1)

        # (batch,)
        len_p = get_lengths_from_binary_sequence_mask(mask_p)
        len_h = get_lengths_from_binary_sequence_mask(mask_h)

        # (batch, 1, hidden_dim)
        last_token_p = torch.clamp(len_p - 1, min=0).view(-1, 1, 1)
        last_token_h = torch.clamp(len_h - 1, min=0).view(-1, 1, 1)

        # (batch, 1, hidden_dim)
        last_token_p = last_token_p.expand(-1, 1, self.hidden_dim)
        last_token_h = last_token_h.expand(-1, 1, self.hidden_dim)

        # array to keep the matching vectors for premise and hypothesis
        mv_p, mv_h = [], []

        # First calculate the cosine similarities between each forward
        # (or backward) contextual embedding and every forward (or backward)
        # contextual embedding of the other sentence.

        # (batch, seq_len1, seq_len2)
        cosine_fw = cosine_pairwise(context_p_fw, context_h_fw)
        cosine_bw = cosine_pairwise(context_p_bw, context_h_bw)

        # (batch, seq_len1, 1)
        cosine_fw_max_p, _ = cosine_fw.max(dim=2, keepdim=True)
        cosine_bw_max_p, _ = cosine_bw.max(dim=2, keepdim=True)
        cosine_fw_mean_p = cosine_fw.mean(dim=2, keepdim=True)
        cosine_bw_mean_p = cosine_bw.mean(dim=2, keepdim=True)

        # (batch, seq_len2, 1)
        cosine_fw_max_h, _ = cosine_fw.max(dim=1, keepdim=True)
        cosine_bw_max_h, _ = cosine_bw.max(dim=1, keepdim=True)
        cosine_fw_mean_h = cosine_fw.mean(dim=1, keepdim=True)
        cosine_bw_mean_h = cosine_bw.mean(dim=1, keepdim=True)
        cosine_fw_max_h = cosine_fw_max_h.permute(0, 2, 1)
        cosine_bw_max_h = cosine_bw_max_h.permute(0, 2, 1)
        cosine_fw_mean_h = cosine_fw_mean_h.permute(0, 2, 1)
        cosine_bw_mean_h = cosine_bw_mean_h.permute(0, 2, 1)

        mv_p.extend([cosine_fw_max_p, cosine_fw_mean_p, cosine_bw_max_p, cosine_bw_mean_p])
        mv_h.extend([cosine_fw_max_h, cosine_fw_mean_h, cosine_bw_max_h, cosine_bw_mean_h])

        mv_idx = 0

        # 1. Full-Matching
        # Each time step of forward (or backward) contextual embedding of one sentence
        # is compared with the last time step of the forward (or backward)
        # contextual embedding of the other sentence
        if not self.wo_full_match:

            # (batch, 1, hidden_size)
            context_p_fw_last = context_p_fw.gather(1, last_token_p)
            context_h_fw_last = context_h_fw.gather(1, last_token_h)
            context_p_bw_last = context_p_bw[:, 0:1, :]
            context_h_bw_last = context_h_bw[:, 0:1, :]

            # (batch, seq_len, num_perspective)
            mv_p_full_fw = mpm(context_p_fw, context_h_fw_last.expand(-1, seq_len_p, -1), self.params[mv_idx])
            mv_p_full_bw = mpm(context_p_bw, context_h_bw_last.expand(-1, seq_len_p, -1), self.params[mv_idx + 1])
            mv_h_full_fw = mpm(context_h_fw, context_p_fw_last.expand(-1, seq_len_h, -1), self.params[mv_idx])
            mv_h_full_bw = mpm(context_h_bw, context_p_bw_last.expand(-1, seq_len_h, -1), self.params[mv_idx + 1])

            mv_p.extend(mv_p_full_fw + mv_p_full_bw)
            mv_h.extend(mv_h_full_fw + mv_h_full_bw)

            mv_idx += 2

        # 2. Maxpooling-Matching
        # Each time step of forward (or backward) contextual embedding of one sentence
        # is compared with every time step of the forward (or backward)
        # contextual embedding of the other sentence, and only the max value of each
        # dimension is retained.
        if not self.wo_maxpool_match:
            # (batch, seq_len1, seq_len2, num_perspective)
            mv_max_fw = mpm_pairwise(context_p_fw, context_h_fw, self.params[mv_idx])
            mv_max_bw = mpm_pairwise(context_p_bw, context_h_bw, self.params[mv_idx + 1])

            # (batch, seq_len, num_perspective)
            mv_p_max_fw, _ = mv_max_fw.max(dim=2)
            mv_p_mean_fw = mv_max_fw.mean(dim=2)
            mv_p_max_bw, _ = mv_max_bw.max(dim=2)
            mv_p_mean_bw = mv_max_bw.mean(dim=2)
            mv_h_max_fw, _ = mv_max_fw.max(dim=1)
            mv_h_mean_fw = mv_max_fw.mean(dim=1)
            mv_h_max_bw, _ = mv_max_bw.max(dim=1)
            mv_h_mean_bw = mv_max_bw.mean(dim=1)
            mv_p.extend([mv_p_max_fw, mv_p_mean_fw, mv_p_max_bw, mv_p_mean_bw])
            mv_h.extend([mv_h_max_fw, mv_h_mean_fw, mv_h_max_bw, mv_h_mean_bw])

            mv_idx += 2

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
        att_h_fw = context_h_fw.unsqueeze(1) * cosine_fw.unsqueeze(3)
        att_h_bw = context_h_bw.unsqueeze(1) * cosine_bw.unsqueeze(3)

        # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_p_fw = context_p_fw.unsqueeze(2) * cosine_fw.unsqueeze(3)
        att_p_bw = context_p_bw.unsqueeze(2) * cosine_bw.unsqueeze(3)

        if not self.wo_attentive_match:
            # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) ->
            # (batch, seq_len1, hidden_size)
            att_mean_h_fw = div_safe(att_h_fw.sum(dim=2), cosine_fw.sum(dim=2, keepdim=True))
            att_mean_h_bw = div_safe(att_h_bw.sum(dim=2), cosine_bw.sum(dim=2, keepdim=True))

            # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) ->
            # (batch, seq_len2, hidden_size)
            att_mean_p_fw = div_safe(att_p_fw.sum(dim=1), cosine_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
            att_mean_p_bw = div_safe(att_p_bw.sum(dim=1), cosine_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))

            # (batch, seq_len, num_perspective)
            mv_p_att_mean_fw = mpm(context_p_fw, att_mean_h_fw, self.params[mv_idx])
            mv_p_att_mean_bw = mpm(context_p_bw, att_mean_h_bw, self.params[mv_idx + 1])
            mv_h_att_mean_fw = mpm(context_h_fw, att_mean_p_fw, self.params[mv_idx])
            mv_h_att_mean_bw = mpm(context_h_bw, att_mean_p_bw, self.params[mv_idx + 1])
            mv_p.extend(mv_p_att_mean_fw + mv_p_att_mean_bw)
            mv_h.extend(mv_h_att_mean_fw + mv_h_att_mean_bw)

            mv_idx += 2

        # 4. Max-Attentive-Matching
        # Pick the contextual embeddings with the highest cosine similarity as the attentive
        # vector, and match each forward (or backward) contextual embedding with its
        # corresponding attentive vector.
        if not self.wo_max_attentive_match:
            # (batch, seq_len1, hidden_size)
            att_max_h_fw, _ = att_h_fw.max(dim=2)
            att_max_h_bw, _ = att_h_bw.max(dim=2)
            # (batch, seq_len2, hidden_size)
            att_max_p_fw, _ = att_p_fw.max(dim=1)
            att_max_p_bw, _ = att_p_bw.max(dim=1)

            # (batch, seq_len, num_perspective)
            mv_p_att_max_fw = mpm(context_p_fw, att_max_h_fw, self.params[mv_idx])
            mv_p_att_max_bw = mpm(context_p_bw, att_max_h_bw, self.params[mv_idx + 1])
            mv_h_att_max_fw = mpm(context_h_fw, att_max_p_fw, self.params[mv_idx])
            mv_h_att_max_bw = mpm(context_h_bw, att_max_p_bw, self.params[mv_idx + 1])

            mv_p.extend(mv_p_att_max_fw + mv_p_att_max_bw)
            mv_h.extend(mv_h_att_max_fw + mv_h_att_max_bw)

            mv_idx += 2

        # Lastly, concatenate the four matching results
        # (batch, seq_len, num_perspective * num_matching)
        mv_p = torch.cat(mv_p, dim=2)
        mv_h = torch.cat(mv_h, dim=2)

        return mv_p, mv_h


if __name__ == "__main__":

    from allennlp.common import Params

    torch.set_printoptions(linewidth=150, edgeitems=3)
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

    ml = MatchingLayer.from_params(Params({"num_perspective": num_perspective}))

    vecs_p, vecs_h = ml(test1, mask1, test2, mask2)

    assert vecs_p.size() == torch.Size([batch, len1, 4 + 6 * (num_perspective + 1) + 4 * num_perspective])
    assert vecs_h.size() == torch.Size([batch, len2, 4 + 6 * (num_perspective + 1) + 4 * num_perspective])

    result_len_p = get_lengths_from_binary_sequence_mask(vecs_p > 0.0)
    result_len_h = get_lengths_from_binary_sequence_mask(vecs_h > 0.0)

    print("vecs_p", vecs_p.shape, vecs_p)
    print("vecs_h", vecs_h.shape, vecs_h)