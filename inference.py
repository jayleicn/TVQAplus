import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.stage import STAGE
from tvqa_dataset import TVQADataset, pad_collate, prepare_inputs
from config import TestOptions
from utils import save_json_pretty, merge_dicts, save_json


def find_max_pair(p1, p2):
    """ Find (k1, k2) where k1 <= k2 with the maximum value of p1[k1] * p2[k2]
    Args:
        p1: a list of probablity for start_idx
        p2: a list of probablity for end_idx
    Returns:
        best_span: (st_idx, ed_idx)
        max_value: probability of this pair being correct
    """
    max_val = 0
    best_span = (0, 1)
    argmax_k1 = 0
    for i in range(len(p1)):
        val1 = p1[argmax_k1]
        if val1 < p1[i]:
            argmax_k1 = i
            val1 = p1[i]

        val2 = p2[i]
        if val1 * val2 > max_val:
            best_span = (argmax_k1, i)
            max_val = val1 * val2
    return best_span, float(max_val)


def inference(opt, dset, model):
    dset.set_mode(opt.mode)
    data_loader = DataLoader(dset, batch_size=opt.test_bsz, shuffle=False, collate_fn=pad_collate)

    predictions = dict(ts_answer={}, raw_bbox=[])
    max_len_dict = dict(
        max_sub_l=opt.max_sub_l,
        max_vid_l=opt.max_vid_l,
        max_vcpt_l=opt.max_vcpt_l,
        max_qa_l=opt.max_qa_l,
    )
    for valid_idx, batch in tqdm(enumerate(data_loader)):
        model_inputs, targets, qids = prepare_inputs(batch, max_len_dict=max_len_dict, device=opt.device)
        model_inputs.use_hard_negatives = False
        model_inputs.eval_object_word_ids = dset.eval_object_word_ids  # so we know which words need boxes.

        inference_outputs = model(model_inputs)
        # predicted answers
        pred_ids = inference_outputs["answer"].data.max(1)[1]

        # predicted regions
        if inference_outputs["att_predictions"]:
            predictions["raw_bbox"] += inference_outputs["att_predictions"]

        temporal_predictions = inference_outputs["t_scores"]
        for qid, pred_a_idx, temporal_score_st, temporal_score_ed, img_indices in \
                zip(qids, pred_ids.tolist(),
                    temporal_predictions[:, :, :, 0],
                    temporal_predictions[:, :, :, 1],
                    model_inputs["image_indices"]):
            offset = (img_indices[0] % 6) / 3
            (st, ed), _ = find_max_pair(temporal_score_st[pred_a_idx].cpu().numpy().tolist(),
                                        temporal_score_ed[pred_a_idx].cpu().numpy().tolist())
            # [[st, ed], pred_ans_idx], note that [st, ed] is associated with the predicted answer.
            predictions["ts_answer"][str(qid)] = [[st * 2 + offset, (ed + 1) * 2 + offset], int(pred_a_idx)]
        if opt.debug:
            break
    return predictions


def main_inference():
    print("Loading config...")
    opt = TestOptions().parse()
    print("Loading dataset...")
    dset = TVQADataset(opt, mode=opt.mode)
    print("Loading model...")
    model = STAGE(opt)
    model.to(opt.device)
    cudnn.benchmark = True
    strict_mode = not opt.no_strict
    model_path = os.path.join("results", opt.model_dir, "best_valid.pth")
    model.load_state_dict(torch.load(model_path), strict=strict_mode)
    model.eval()
    model.inference_mode = True
    torch.set_grad_enabled(False)
    print("Evaluation Starts:\n")
    predictions = inference(opt, dset, model)
    print("predictions {}".format(predictions.keys()))
    pred_path = model_path.replace("best_valid.pth",
                                   "{}_inference_predictions.json".format(opt.mode))
    save_json(predictions, pred_path)


if __name__ == "__main__":
    main_inference()
