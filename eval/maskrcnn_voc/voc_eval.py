from __future__ import division
__author__ = "Jie Lei"
# ref https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/
#            maskrcnn_benchmark/data/datasets/evaluation/voc/voc_eval.py
# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)

from collections import defaultdict
import numpy as np
from bounding_box import BoxList
from boxlist_ops import boxlist_iou


def eval_detection_voc(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Evaluate on voc dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec, n_tp, n_fp, n_pos = calc_detection_voc_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)
    prec = {k: v.tolist() for k, v in prec.items()}
    rec = {k: v.tolist() for k, v in rec.items()}
    res = [{"ap": ap[k], "class_id": k, "precisions": prec[k], "recalls": rec[k],
            "n_tp": n_tp[k], "n_fp": n_fp[k], "n_positives": n_pos[k]} for k in ap.keys()]
    return res, np.nanmean(ap.values())


def calc_detection_voc_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    gt_labels = []
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        # pred_bbox = pred_boxlist.bbox.numpy()
        # pred_label = pred_boxlist.get_field("labels").numpy()
        # pred_score = pred_boxlist.get_field("scores").numpy()
        # gt_bbox = gt_boxlist.bbox.numpy()
        # gt_label = gt_boxlist.get_field("labels").numpy()
        # gt_difficult = gt_boxlist.get_field("difficult").numpy()
        pred_bbox = pred_boxlist.bbox
        pred_label = pred_boxlist.get_field("labels")
        pred_score = pred_boxlist.get_field("scores")
        gt_bbox = gt_boxlist.bbox
        gt_label = gt_boxlist.get_field("labels")
        gt_difficult = gt_boxlist.get_field("difficult")
        gt_labels.append(gt_label)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            # iou = boxlist_iou(
            #     BoxList(pred_bbox_l, gt_boxlist.size),
            #     BoxList(gt_bbox_l, gt_boxlist.size),
            # ).numpy()
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            )

            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    # n_fg_class = max(n_pos.keys()) + 1
    # prec = [None] * n_fg_class
    # rec = [None] * n_fg_class
    gt_labels = np.concatenate(gt_labels)
    n_pos = {}
    for l in np.unique(gt_labels.astype(int)):
        m1 = np.sum(gt_labels == l)
        m2 = np.sum(gt_labels.astype(int) == l)
        if m1 != m2:
            print(m1, m2)
        n_pos[l] = m2

    prec = {}
    rec = {}
    n_fp = {}
    n_tp = {}

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        n_tp[l] = np.sum(match_l == 1)
        n_fp[l] = np.sum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
        else:
            rec[l] = None

    return prec, rec, n_tp, n_fp, n_pos


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (dict of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (dict of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    # n_fg_class = len(prec)
    ap = {}  # np.empty(n_fg_class)
    for l in prec.keys():
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def test_precision():
    recall = [
        0.0,
        0.1,
        0.2,
        0.2,
        0.2,
        0.2,
        0.3,
        0.4,
        0.4,
        0.4,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5
    ]
    precision = [
        0.0,
        0.5,
        0.6666666666666666,
        0.5,
        0.4,
        0.3333333333333333,
        0.42857142857142855,
        0.5,
        0.4444444444444444,
        0.4,
        0.45454545454545453,
        0.4166666666666667,
        0.38461538461538464,
        0.35714285714285715,
        0.3333333333333333,
        0.3125,
        0.29411764705882354
    ]

    def mrcnn_calc_ap(rec, prec):
        # correct AP calculation
        # first append sentinel values at the end
        mpre = np.concatenate(([0], np.nan_to_num(prec), [0]))
        mrec = np.concatenate(([0], rec, [1]))

        mpre = np.maximum.accumulate(mpre[::-1])[::-1]

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap, mpre[0:-1], mrec[0:-1]

    def CalculateAveragePrecision(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return ap, mpre[0:-1], mrec[0:-1]

    mrcn_res = mrcnn_calc_ap(recall, precision)
    cus_res = CalculateAveragePrecision(recall, precision)
    print(mrcn_res)
    print(cus_res)


def test_diff(path1, path2, key="total positives"):
    metrics1 = load_json(path1)
    metrics2 = load_json(path2)
    cls2npos1 = {e["class"]: e[key] for e in metrics1}
    cls2npos2 = {e["class"]: e[key] for e in metrics2}
    diff = {}
    for k in cls2npos1:
        diff[k] = cls2npos1[k] - cls2npos2[k]
    diff_counter = Counter(diff.values())
    return diff, diff_counter


if __name__ == '__main__':
    test_precision()

