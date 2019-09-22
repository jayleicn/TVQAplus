__author__ = "Jie Lei"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle


def save_pickle(data, data_path):
    with open(data_path, "w") as f:
        pickle.dump(data, f)


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


class NormalizeScale(nn.Module):
    def __init__(self, dim, num_additional_dims=1, init_norm=20):
        super(NormalizeScale, self).__init__()
        self.init_norm = init_norm
        dims = [1] * num_additional_dims + [dim]
        self.weight = nn.Parameter(torch.ones(dims) * init_norm)

    def forward(self, bottom):
        # input is variable (*, dim)
        bottom_normalized = nn.functional.normalize(bottom, p=2, dim=1)
        bottom_normalized_scaled = bottom_normalized * self.weight
        return bottom_normalized_scaled


def compute_temporal_iou(pred, gt):
    """ compute intersection-over-union along temporal axis
    Ref: https://github.com/LisaAnne/LocalizingMoments/blob/master/utils/eval.py
    Args:
        pred: [st (float), ed (float)]
        gt: [st (float), ed (float)]
    Returns:
        iou (float):
    """
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])  # not the correct union though
    if union == 0:
        return 0
    else:
        return 1.0 * intersection / union


def get_high_iou_sapns(gt_ts_list, pred_ts_list, iou_thd=0.5, add_gt=True):
    """ Note
    Args:
        gt_ts_list: N * (st, ed)
        pred_ts_list: N * [(st_idx, ed_idx, confidence), ...]
        iou_thd (float):
        add_gt (bool):
    Returns:

    """
    spans = []  # container for both pred and gt (st, ed),
    for idx, (gt_ts, pred_ts_sublist) in enumerate(zip(gt_ts_list, pred_ts_list)):
        if add_gt:
            cur_spans = [gt_ts]
        else:
            cur_spans = []
        for pred_ts in pred_ts_sublist:
            pred_ts = pred_ts[:2]  # (st, ed)
            if compute_temporal_iou(pred_ts, gt_ts) >= iou_thd:
                cur_spans.append(pred_ts)
        spans.append(cur_spans)
    return spans  # N * [(st, ed), ...]


def expand_span(span, expand_length=2):
    """
    Args:
        span (list): [st, ed]
        expand_length (int): length to add on the two sides

    Returns:
        expanded_span (list): [max(0, st-expand_length), ed + expand_length]
            Only use the span for indexing, no need to worry the case where
            (ed + expand_length) >= max_length.
    """
    return [max(0, span[0] - expand_length), span[1] + expand_length]


def find_max_triples(p1, p2, topN=5, prob_thd=None):
    """ Find a list of (k1, k2) where k1 >= k2 with the maximum values of p1[k1] * p2[k2]
    Args:
        p1 (torch.CudaTensor): (N, L) batched start_idx probabilities
        p2 (torch.CudaTensor): (N, L) batched end_idx probabilities
        topN (int): return topN pairs with highest values
        prob_thd (float):
    Returns:
        batched_sorted_triple: N * [(st_idx, ed_idx, confidence), ...]
    """
    product = torch.bmm(p1.unsqueeze(2), p2.unsqueeze(1))  # (N, L, L), end_idx >= start_idx
    upper_product = torch.stack([torch.triu(p) for p in product]
                                ).data.cpu().numpy()  # (N, L, L) the lower part becomes zeros
    batched_sorted_triple = []
    for idx, e in enumerate(upper_product):
        sorted_triple = topN_array_2d(e, topN=topN)
        if prob_thd is not None:
            sorted_triple = [t for t in sorted_triple if t[2] >= prob_thd]
        batched_sorted_triple.append(sorted_triple)
    return batched_sorted_triple


def topN_array_2d(array_2d, topN=None):
    """ Get topN indices and values of a 2d array, return a tuple of indices and their values,
    ranked by the value
    """
    row_indices, column_indices = np.unravel_index(np.argsort(array_2d, axis=None), array_2d.shape)
    row_indices = row_indices[::-1][:topN]
    column_indices = column_indices[::-1][:topN]
    sorted_values = array_2d[row_indices, column_indices]
    sorted_triples = zip(row_indices, column_indices, sorted_values)
    return sorted_triples
