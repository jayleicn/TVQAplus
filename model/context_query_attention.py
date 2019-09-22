import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredAttention(nn.Module):
    """Use each word in context to attend to words in query.
    In my case, context is question-answer, query is the object-level
    features in an image.

    Note the values in S are cosine similarity scores, and are in [-1, 1]
    They are scaled before softmax to make sure the maximum value could
    get very high probability.
    S_ = F.softmax(S * self.scale, dim=-1)

    Consider softmax function f(m) = exp(m) / [24 * exp(-m) + exp(m)]
    If not scaled, S * scale \in [-100, 100], the weight the maximum value could only get is
    exp(1) / [24 * exp(-1) + exp(1)] = 0.04 .
    When set the scale = 100, S * scale \in [-100, 100]
    exp(100) / [24 * exp(-100) + exp(100)] = 0.9976
    """
    def __init__(self, dropout=0.1, scale=100, add_void=False):
        """
        Args:
            dropout:
            scale:
            add_void:
        """
        super(StructuredAttention, self).__init__()
        self.dropout = dropout
        self.scale = scale
        self.add_void = add_void

    def forward(self, C, Q, c_mask, q_mask, noun_mask=None, void_vector=None):
        """
        match the dim of '*', singlton is allowed
        Args:
            C: (N, 5, Li, Lqa, D)
            Q: (N, 1, Li, Lr, D)
            c_mask: (N, 5, Li, Lqa)
            q_mask: (N, 1, Li, Lr)
            noun_mask: (N, 5, Lqa) , where 1 indicate the current position is a noun
               or (N, 5, Li, Lqa), where each entry is the probability of the current
               image being a positive bag for the word
            void_vector: (D, )
        Returns:
            (N, *, Lc, D)
        """
        bsz, _, num_img, num_region, hsz = Q.shape
        if void_vector is not None:
            num_void = len(void_vector)
            Q_void = void_vector.view(1, 1, 1, num_void, hsz).repeat(bsz, 1, num_img, 1, 1)
            Q = torch.cat([Q, Q_void], dim=-2)  # (N, 1, Li, Lr+num_void, D)
            q_mask_void = q_mask.new_ones(bsz, 1, num_img, num_void)  # ones
            q_mask = torch.cat([q_mask, q_mask_void], dim=-1)  # (N, 1, Li, Lr+num_void)

        S, S_mask = self.similarity(C, Q, c_mask, q_mask)  # (N, 5, Li, Lqa, Lr+num_void)
        S_ = F.softmax(S * self.scale, dim=-1)
        # (N, 5, Li, Lqa, Lr+1) # the weight of each query word to a given context word
        S_ = S_ * S_mask  # for columns that are all padded elements

        if noun_mask is not None:
            if len(noun_mask.shape) == 3:
                bsz, num_qa, lqa = noun_mask.shape
                S_ = S_ * noun_mask.view(bsz, num_qa, 1, lqa, 1)
            elif len(noun_mask.shape) == 4:
                S_ = S_ * noun_mask.unsqueeze(-1)
            else:
                raise NotImplementedError

        if void_vector is not None:
            if self.add_void:
                A = torch.matmul(S_, Q)  # (N, 5, Li, Lqa, D)
                S, S_mask, S_ = S[:, :, :, :, :-num_void], S_mask[:, :, :, :, :-num_void], S_[:, :, :, :, :-num_void]
            else:
                S, S_mask, S_ = S[:, :, :, :, :-num_void], S_mask[:, :, :, :, :-num_void], S_[:, :, :, :, :-num_void]
                Q = Q[:, :, :, :-num_void, :]  # (N, 1, Li, Lr, D)
                A = torch.matmul(S_, Q)  # (N, 5, Li, Lqa, D)
        else:
            A = torch.matmul(S_, Q)  # (N, 5, Li, Lqa, D)
        return A, S, S_mask, S_

    def similarity(self, C, Q, c_mask, q_mask):
        """
        word2word dot-product similarity
        Args:
            C: (N, 5, Li, Lqa, D)
            Q: (N, 1, Li, Lr, D)
            c_mask: (N, 5, Li, Lqa)
            q_mask: (N, 1, Li, Lr)
        Returns:
            (N, *, Lc, Lq)
        """
        C = F.dropout(F.normalize(C, p=2, dim=-1), p=self.dropout, training=self.training)
        Q = F.dropout(F.normalize(Q, p=2, dim=-1), p=self.dropout, training=self.training)

        S_mask = torch.matmul(c_mask.unsqueeze(-1), q_mask.unsqueeze(-2))  # (N, 5, Li, Lqa, Lr)
        S = torch.matmul(C, Q.transpose(-2, -1))  # (N, 5, Li, Lqa, Lr)
        masked_S = S - 1e10*(1 - S_mask)  # (N, 5, Li, Lqa, Lr)
        return masked_S, S_mask


class ContextQueryAttention(nn.Module):
    """
    sub-a attention
    """
    def __init__(self):
        super(ContextQueryAttention, self).__init__()

    def forward(self, C, Q, c_mask, q_mask):
        """
        match the dim of '*', singlton is allowed
        :param C: (N, *, Lc, D)
        :param Q: (N, *, Lq, D)
        :param c_mask: (N, *, Lc)
        :param q_mask: (N, *, Lq)
        :return: (N, Lc, D) and (N, Lq, D)
        """

        S = self.similarity(C, Q, c_mask, q_mask)  # (N, *, Lc, Lq)
        S_ = F.softmax(S, dim=-1)  # (N, *, Lc, Lq)
        A = torch.matmul(S_, Q)  # (N, *, Lc, D)
        return A

    def similarity(self, C, Q, c_mask, q_mask):
        """
        word2word dot-product similarity
        :param C: (N, *, Lc, D)
        :param Q: (N, *, Lq, D)
        :param c_mask: (N, *, Lc)
        :param q_mask: (N, *, Lq)
        :return: (N, *, Lc, Lq)
        """
        C = F.dropout(C, p=0.1, training=self.training)
        Q = F.dropout(Q, p=0.1, training=self.training)
        hsz_root = math.sqrt(C.shape[-1])

        S_mask = torch.matmul(c_mask.unsqueeze(-1), q_mask.unsqueeze(-2))  # (N, *, Lc, Lq)
        S = torch.matmul(C, Q.transpose(-2, -1)) / hsz_root  # (N, *, Lc, Lq)
        masked_S = S - 1e10*(1 - S_mask)  # (N, *, Lc, Lq)
        return masked_S


def test():
    # (N, *, D, Lc)
    c2q = ContextQueryAttention()
    hsz = 128
    bsz = 10
    lc = 20
    lq = 10
    context = torch.randn(bsz, hsz, lc).float()
    context_mask = torch.ones(bsz, lc).float()
    query = torch.randn(bsz, hsz, lq).float()
    query_mask = torch.ones(bsz, lq).float()
    a, b = c2q(context, query, context_mask, query_mask)
    print("input size", context.shape, context_mask.shape, query.shape, query_mask.shape)
    print("output size", a.shape, b.shape)


if __name__ == '__main__':
    test()

