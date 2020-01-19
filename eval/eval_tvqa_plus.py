import os
import torch
import numpy as np
from collections import defaultdict
from maskrcnn_voc import *
from utils import load_json, save_json_pretty, merge_dicts, save_json


def clean_label(label_str):
    return label_str.replace(u"\u2019", "'").replace(u"\u2018", "'").lower()


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


def compute_temporal_metrics(pred_dict, gt_dict):
    """ evaluate TVQA temporal localization, QA Acc., ASA.
    Args:
        pred_dict (dict): predicted moments (with start and end timestamps)
            {"qid": [(st (float), ed (float)), ans_idx(int)], ...}
        gt_dict (dict): ground-truth moments (with start and end timestamps)
            {"qid": [(st (float), ed (float)), ans_idx(int)],...}
    Returns:
        recalls (list): Recall at different iou thresholds, from [0.1-0.9]
        miou (float): mean intersection-over-union
        iou_array (list of float):
    """
    keys = sorted(pred_dict.keys())
    gt_key_type = type(gt_dict.keys()[0])  # in case the two dicts has differnt key types

    pred_ts = [pred_dict[k][0] for k in keys]  # (N, 2) # [st, ed]
    gt_ts = [gt_dict[gt_key_type(k)][0] for k in keys]  # (N, 2) # [st, ed]
    iou_array = []
    for pred, gt in zip(pred_ts, gt_ts):
        iou_array.append(compute_temporal_iou(pred, gt))
    iou_array = np.array(iou_array)  # (N, ) float

    pred_ans = np.array([pred_dict[k][1] for k in keys])  # (N, )
    gt_ans = np.array([gt_dict[gt_key_type(k)][1] for k in keys])  # (N, )
    answer_mask = pred_ans == gt_ans  # (N, ) bool

    iou_thds = np.arange(0.1, 1, 0.1)
    res = {}
    for iou_thd in iou_thds:
        res["R@{:.2f}".format(iou_thd)] = 1.0 * np.sum(iou_array >= iou_thd) / len(iou_array)

    # temporal mIoU
    res["miou"] = 1.0 * np.sum(iou_array) / len(iou_array)
    # ASA
    res["ans_span_joint_acc@.5"] = 1.0 * np.sum(answer_mask * (iou_array >= 0.5)) / len(answer_mask)
    # QA Acc.
    res["qa_acc"] = 1.0 * np.sum(answer_mask) / len(answer_mask)
    return res


def transform_metrics_per_class(metrics_per_class, idx2word):
    """from list to dict, add one entry label"""
    transformed_metrics_per_class = {}
    for e in metrics_per_class:
        e["label"] = idx2word[e["class_id"]]
        transformed_metrics_per_class[e["label"]] = e
    return transformed_metrics_per_class


def compute_att_metrics_using_maskrcnn_voc(pred_im2boxes, gt_im2boxes, word2idx):
    """
    Args:
        pred_im2boxes: {qid: {frm_id: list([label, box])}}
        gt_im2boxes: {qid: {frm_id: list([label, box])}}
        word2idx:
    """
    # re-organzie the predictions
    def get_boxes_by_image(raw_boxes):
        """{qid: {frm_id: list([label, box])}}"""
        boxes_by_image = defaultdict(list)
        for qid, qid_data in raw_boxes.items():
            for frm_id, frm_data in qid_data.items():
                vid_name = frm_data["vid_name"]
                img_name = "{}_{}_{:05d}".format(vid_name, int(qid), int(frm_id))
                boxes_by_image[img_name] = frm_data["boxes"]
        return boxes_by_image

    def get_boxlist_by_image(boxes_by_image, w2i, add_difficult=False, rm_unk=True):
        """whether to add additional entry in BoxList, add_difficult"""
        boxlist_by_image = {}
        label_vocab = []
        for img_name, v in boxes_by_image.items():
            labels = [w2i[e[0]] if e[0] in w2i else w2i["<unk>"] for e in v]
            label_vocab.extend([e[0] for e in v])
            scores = [e[1] for e in v]
            boxes = [e[2] for e in v]
            if rm_unk:
                non_unk_label_indices = [idx for idx, e in enumerate(labels) if int(e) != w2i["<unk>"]]
                labels = [labels[idx] for idx in non_unk_label_indices]
                scores = [scores[idx] for idx in non_unk_label_indices]
                boxes = [boxes[idx] for idx in non_unk_label_indices]
                if len(boxes) == 0:
                    continue
            boxlist_by_image[img_name] = BoxList(boxes, image_size=(640, 360), mode="xyxy")
            boxlist_by_image[img_name].add_field("labels", torch.Tensor(labels))
            boxlist_by_image[img_name].add_field("scores", torch.Tensor(scores))
            if add_difficult:
                boxlist_by_image[img_name].add_field("difficult", torch.Tensor([0] * len(labels)))
        return boxlist_by_image, list(set(label_vocab))

    # pred_im2boxes = get_boxes_by_image(pred_boxes)
    pred_im2boxlist_dict, pred_label_vocab = get_boxlist_by_image(pred_im2boxes, word2idx, add_difficult=False)
    gt_im2boxlist_dict, gt_label_vocab = get_boxlist_by_image(gt_im2boxes, word2idx, add_difficult=True)

    empty_pred_boxlist = BoxList([[0, 0, 0, 0]], image_size=(640, 360), mode="xyxy")
    empty_pred_boxlist.add_field("labels", torch.Tensor([0]))
    empty_pred_boxlist.add_field("scores", torch.Tensor([0]))

    gt_boxlists = []  # list(BoxList)
    pred_boxlists = []  # list(BoxList)
    for img_k, gt_bl in gt_im2boxlist_dict.items():
        gt_boxlists.append(gt_bl)
        if img_k not in pred_im2boxlist_dict:
            pred_boxlists.append(empty_pred_boxlist)
        else:
            pred_boxlists.append(pred_im2boxlist_dict[img_k])

    print("Start TVQAplus evaluation...")
    metrics_per_class, mAP = eval_detection_voc(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False)
    idx2word = {idx: word for word, idx in word2idx.items()}
    metrics_per_class = transform_metrics_per_class(metrics_per_class, idx2word)
    return {"metrics_per_class": metrics_per_class, "overall_map": mAP}


def load_tvqa_plus_annotation(anno_path):
    raw_annotation = load_json(anno_path)
    gt_box_info = defaultdict(list)
    gt_ts_answer = defaultdict(dict)
    for e in raw_annotation:
        qid = e["qid"]
        vid_name = e["vid_name"]
        # {"qid": [(st (float), ed (float)), ans_idx(int)], ...}
        gt_ts_answer[qid] = [e["ts"], int(e["answer_idx"])]
        for frm_id, box_info_list in e["bbox"].items():
            img_name = "{}_{}_{:05d}".format(vid_name, int(qid), int(frm_id))
            for single_box in box_info_list:
                # [label, score=1 (fake score), box_coordinates (xyxy)]
                reformatted_single_box = [clean_label(single_box["label"]), 1,
                                          [single_box["left"], single_box["top"],
                                           single_box["left"]+single_box["width"],
                                           single_box["top"]+single_box["height"]]]
                gt_box_info[img_name].append(reformatted_single_box)
    annotation = dict(
        ts_answer=gt_ts_answer,
        bbox=gt_box_info
    )
    return annotation


def load_predictions(pred_path, gt_path, w2i_path):
    """gt_path stores ground truth data, here used to reformat the predictions"""
    raw_preds = load_json(pred_path)
    gt_data = load_json(gt_path)
    word2idx = load_json(w2i_path)
    idx2word = {i: w for w, i in word2idx.items()}
    qid2ans = {int(e["qid"]): int(e["answer_idx"]) for e in gt_data}
    qid2bbox = {int(e["qid"]): e["bbox"] for e in gt_data}
    bbox_preds = dict()
    for e in raw_preds["raw_bbox"]:
        qid = None
        for i in range(5):  # n_answer == 5
            if len(e[str(i)]) > 0:
                qid = e[str(i)][0]["qid"]
        assert qid is not None
        ans_idx = qid2ans[int(qid)]
        cur_gt_bbox = qid2bbox[int(qid)]
        cur_correct_bbox_preds = e[str(ans_idx)]
        key_template = "{vid_name}_{qid}_{img_idx:05d}"
        for p in cur_correct_bbox_preds:
            annotated_word_ids = [word2idx[clean_label(b["label"])] if clean_label(b["label"]) in word2idx
                                  else word2idx["<unk>"] for b in cur_gt_bbox[str(p["img_idx"])]]
            collected_bbox = []
            for idx, b in enumerate(p["bbox"]):
                if p["word"] in annotated_word_ids:
                    collected_bbox.append([idx2word[p["word"]], float(p["pred"][idx]), b])
            key_str = key_template.format(vid_name=p["vid_name"], qid=qid, img_idx=p["img_idx"])
            if key_str not in bbox_preds:
                bbox_preds[key_str] = []
            bbox_preds[key_str].extend(collected_bbox)

    preds = dict(ts_answer=raw_preds["ts_answer"], bbox=bbox_preds)
    return preds


def main_eval():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, default="data/tvqa_plus_val.json",
                        help="ground-truth json file path")
    parser.add_argument("--pred_path", type=str,
                        help="input prediction json file path, the same format as the results "
                             "returned by load_tvqa_plus_annotation func")
    parser.add_argument("--word2idx_path", type=str, default="data/word2idx.json",
                        help="word2idx json file path, provided with the evaluation code")
    parser.add_argument("--output_path", type=str,
                        help="path to store the calculated metrics")
    parser.add_argument("--no_preproc_pred", action="store_true",)
    args = parser.parse_args()

    # Display settings
    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    groundtruth = load_tvqa_plus_annotation(args.gt_path)
    if args.no_preproc_pred:
        prediction = load_json(args.pred_path)
    else:
        prediction = load_predictions(args.pred_path, args.gt_path, args.word2idx_path)
    word2idx = load_json(args.word2idx_path)

    bbox_metrics = compute_att_metrics_using_maskrcnn_voc(prediction["bbox"], groundtruth["bbox"], word2idx)
    temporal_metrics = compute_temporal_metrics(prediction["ts_answer"], groundtruth["ts_answer"])
    all_metrics = merge_dicts([bbox_metrics, temporal_metrics])
    print("QA Acc. {}\nGrd. mAP {}\nTemp. mIoU{}\nASA {}"
          .format(all_metrics["qa_acc"], all_metrics["overall_map"],
                  all_metrics["miou"], all_metrics["ans_span_joint_acc@.5"]))
    if args.output_path:
        save_json_pretty(all_metrics, args.output_path)


if __name__ == '__main__':
    main_eval()

