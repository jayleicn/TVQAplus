import os
import sys
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter

from utils import AverageMeter, count_parameters
from model.stage import STAGE
from tvqa_dataset import TVQADataset, pad_collate, prepare_inputs
from config import BaseOptions


def train(opt, dset, model, criterion, optimizer, epoch, previous_best_acc, use_hard_negatives=False):
    dset.set_mode("train")
    model.train()
    train_loader = DataLoader(dset, batch_size=opt.bsz, shuffle=True,
                              collate_fn=pad_collate, num_workers=opt.num_workers, pin_memory=True)

    train_loss = []
    train_loss_att = []
    train_loss_ts = []
    train_loss_cls = []
    valid_acc_log = ["batch_idx\tacc"]
    train_corrects = []
    torch.set_grad_enabled(True)
    max_len_dict = dict(
        max_sub_l=opt.max_sub_l,
        max_vid_l=opt.max_vid_l,
        max_vcpt_l=opt.max_vcpt_l,
        max_qa_l=opt.max_qa_l,
    )

    # init meters
    dataloading_time = AverageMeter()
    prepare_inputs_time = AverageMeter()
    model_forward_time = AverageMeter()
    model_backward_time = AverageMeter()

    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader)):
        dataloading_time.update(time.time() - timer_dataloading)
        timer_start = time.time()
        model_inputs, _, qids = prepare_inputs(batch, max_len_dict=max_len_dict, device=opt.device)
        prepare_inputs_time.update(time.time() - timer_start)
        model_inputs.use_hard_negatives = use_hard_negatives
        try:
            timer_start = time.time()
            outputs, att_loss, _, temporal_loss, _ = model(model_inputs)
            outputs, targets = outputs
            att_loss = opt.att_weight * att_loss
            temporal_loss = opt.ts_weight * temporal_loss
            cls_loss = criterion(outputs, targets)
            # keep the cls_loss at the same magnitude as only classifying batch_size objects
            cls_loss = cls_loss * (1.0 * len(qids) / len(targets))
            loss = cls_loss + att_loss + temporal_loss
            model_forward_time.update(time.time() - timer_start)
            timer_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()
            model_backward_time.update(time.time() - timer_start)
            # scheduler.step()
            train_loss.append(loss.data.item())
            train_loss_att.append(float(att_loss))
            train_loss_ts.append(float(temporal_loss))
            train_loss_cls.append(cls_loss.item())
            pred_ids = outputs.data.max(1)[1]
            train_corrects += pred_ids.eq(targets.data).tolist()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("WARNING: ran out of memory, skipping batch")
            else:
                print("RuntimeError {}".format(e))
                sys.exit(1)
        if batch_idx % opt.log_freq == 0:
            niter = epoch * len(train_loader) + batch_idx
            if batch_idx == 0:  # do not add to the loss curve, since it only contains a very small
                train_acc = 0
                train_loss = 0
                train_loss_att = 0
                train_loss_ts = 0
                train_loss_cls = 0
            else:
                train_acc = sum(train_corrects) / float(len(train_corrects))
                train_loss = sum(train_loss) / float(len(train_corrects))
                train_loss_att = sum(train_loss_att) / float(len(train_corrects))
                train_loss_cls = sum(train_loss_cls) / float(len(train_corrects))
                train_loss_ts = sum(train_loss_ts) / float(len(train_corrects))
                opt.writer.add_scalar("Train/Acc", train_acc, niter)
                opt.writer.add_scalar("Train/Loss", train_loss, niter)
                opt.writer.add_scalar("Train/Loss_att", train_loss_att, niter)
                opt.writer.add_scalar("Train/Loss_cls", train_loss_cls, niter)
                opt.writer.add_scalar("Train/Loss_ts", train_loss_ts, niter)
            # Test
            valid_acc, valid_loss, qid_corrects = \
                validate(opt, dset, model, criterion, mode="valid", use_hard_negatives=use_hard_negatives)
            opt.writer.add_scalar("Valid/Acc", valid_acc, niter)
            opt.writer.add_scalar("Valid/Loss", valid_loss, niter)

            valid_log_str = "%02d\t%.4f" % (batch_idx, valid_acc)
            valid_acc_log.append(valid_log_str)

            # remember the best acc.
            if valid_acc > previous_best_acc:
                previous_best_acc = valid_acc
                torch.save(model.state_dict(), os.path.join(opt.results_dir, "best_valid.pth"))

            print("Epoch {:02d} [Train] acc {:.4f} loss {:.4f} loss_att {:.4f} loss_ts {:.4f} loss_cls {:.4f}"
                  .format(epoch, train_acc, train_loss, train_loss_att, train_loss_ts, train_loss_cls))

            print("Epoch {:02d} [Val] acc {:.4f} loss {:.4f}"
                  .format(epoch, valid_acc, valid_loss))

            # reset to train
            torch.set_grad_enabled(True)
            model.train()
            dset.set_mode("train")
            train_corrects = []
            train_loss = []
            train_loss_att = []
            train_loss_ts = []
            train_loss_cls = []

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 5:
            print("dataloading_time: max {dataloading_time.max} "
                  "min {dataloading_time.min} avg {dataloading_time.avg}\n"
                  "prepare_inputs_time: max {prepare_inputs_time.max} "
                  "min {prepare_inputs_time.min} avg {prepare_inputs_time.avg}\n"
                  "model_forward_time: max {model_forward_time.max} "
                  "min {model_forward_time.min} avg {model_forward_time.avg}\n"
                  "model_backward_time: max {model_backward_time.max} "
                  "min {model_backward_time.min} avg {model_backward_time.avg}\n"
                  "".format(dataloading_time=dataloading_time, prepare_inputs_time=prepare_inputs_time,
                            model_forward_time=model_forward_time, model_backward_time=model_backward_time))
            break

    # additional log
    with open(os.path.join(opt.results_dir, "valid_acc.log"), "a") as f:
        f.write("\n".join(valid_acc_log) + "\n")

    return previous_best_acc


def validate(opt, dset, model, criterion, mode="valid", use_hard_negatives=False):
    dset.set_mode(mode)
    torch.set_grad_enabled(False)
    model.eval()
    valid_loader = DataLoader(dset, batch_size=opt.test_bsz, shuffle=False,
                              collate_fn=pad_collate, num_workers=opt.num_workers, pin_memory=True)

    valid_qids = []
    valid_loss = []
    valid_corrects = []
    max_len_dict = dict(
        max_sub_l=opt.max_sub_l,
        max_vid_l=opt.max_vid_l,
        max_vcpt_l=opt.max_vcpt_l,
        max_qa_l=opt.max_qa_l,
    )
    for val_idx, batch in enumerate(valid_loader):
        model_inputs, targets, qids = prepare_inputs(batch, max_len_dict=max_len_dict, device=opt.device)
        model_inputs.use_hard_negatives = use_hard_negatives
        outputs, att_loss, _, temporal_loss, _ = model(model_inputs)
        loss = criterion(outputs, targets) + opt.att_weight * att_loss + opt.ts_weight * temporal_loss
        # measure accuracy and record loss
        valid_qids += [int(x) for x in qids]
        valid_loss.append(loss.data.item())
        pred_ids = outputs.data.max(1)[1]
        valid_corrects += pred_ids.eq(targets.data).tolist()
        if opt.debug and val_idx == 20:
            break

    valid_acc = sum(valid_corrects) / float(len(valid_corrects))
    valid_loss = sum(valid_loss) / float(len(valid_corrects))
    qid_corrects = ["%d\t%d" % (a, b) for a, b in zip(valid_qids, valid_corrects)]
    return valid_acc, valid_loss, qid_corrects


def main():
    opt = BaseOptions().parse()
    torch.manual_seed(opt.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(opt.seed)

    writer = SummaryWriter(opt.results_dir)
    opt.writer = writer
    dset = TVQADataset(opt)
    opt.vocab_size = len(dset.word2idx)
    model = STAGE(opt)

    count_parameters(model)

    if opt.device.type == "cuda":
        print("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            print("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU

    criterion = nn.CrossEntropyLoss(reduction="sum").to(opt.device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr,
        weight_decay=opt.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
        verbose=True
    )

    best_acc = 0.
    start_epoch = 0
    early_stopping_cnt = 0
    early_stopping_flag = False
    for epoch in range(start_epoch, opt.n_epoch):
        if not early_stopping_flag:
            use_hard_negatives = epoch + 1 > opt.hard_negative_start  # whether to use hard negative sampling
            niter = epoch * np.ceil(len(dset) / float(opt.bsz))
            opt.writer.add_scalar("learning_rate", float(optimizer.param_groups[0]["lr"]), niter)
            cur_acc = train(opt, dset, model, criterion, optimizer, epoch, best_acc,
                            use_hard_negatives=use_hard_negatives)
            scheduler.step(cur_acc)  # decrease lr when acc is not improving
            # remember best acc
            is_best = cur_acc > best_acc
            best_acc = max(cur_acc, best_acc)
            if not is_best:
                early_stopping_cnt += 1
                if early_stopping_cnt >= opt.max_es_cnt:
                    early_stopping_flag = True
            else:
                early_stopping_cnt = 0
        else:
            print("=> early stop with valid acc %.4f" % best_acc)
            opt.writer.export_scalars_to_json(os.path.join(opt.results_dir, "all_scalars.json"))
            opt.writer.close()
            break  # early stop break

        if opt.debug:
            break

    return opt.results_dir.split("/")[1], opt.debug


if __name__ == "__main__":
    results_dir, debug = main()
