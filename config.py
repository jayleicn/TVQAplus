import os
import time
import torch
import argparse

from utils import mkdirp, load_json, save_json_pretty, make_zipfile


class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        self.parser.add_argument("--debug", action="store_true", help="debug mode, break all loops")
        self.parser.add_argument("--results_dir_base", type=str, default="results/results")
        self.parser.add_argument("--log_freq", type=int, default=800, help="print, save training info")
        self.parser.add_argument("--seed", type=int, default=2018, help="random seed")

        # training config
        self.parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        self.parser.add_argument("--wd", type=float, default=3e-7, help="weight decay")
        self.parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs to run")
        self.parser.add_argument("--max_es_cnt", type=int, default=5, help="number of epochs to early stop")
        self.parser.add_argument("--bsz", type=int, default=16, help="mini-batch size")
        self.parser.add_argument("--test_bsz", type=int, default=16, help="mini-batch size for testing")
        self.parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
        self.parser.add_argument("--device_ids", type=int, nargs="+", default=[0], help="GPU ids to run the job")
        self.parser.add_argument("--num_workers", type=int, default=2,
                                 help="num subprocesses used to load the data, 0: use main process")
        self.parser.add_argument("--t_iter", type=int, default=0,
                                 help="positive integer, indicating #iterations for refine temporal prediction")
        self.parser.add_argument("--t_layer_type", type=str, default="linear", choices=["linear", "conv"],
                                 help="layer type for predicting the localization scores")
        self.parser.add_argument("--extra_span_length", type=int, default=3,
                                 help="expand the boundary of the localized span, "
                                      "by [max(0, pred_st - extra_span_length), pred_ed + extra_span_length]")
        self.parser.add_argument("--ts_weight", type=float, default=0.5, help="temporal loss weight")
        self.parser.add_argument("--add_local", action="store_true",
                                 help="concat local feature with global feature for QA")
        self.parser.add_argument("--input_streams", type=str, nargs="+", default=["sub", "vfeat"],
                                 choices=["vcpt", "sub", "vfeat", "joint_v"],
                                 help="input streams for the model, will use both `vcpt` and `sub` streams")
        self.parser.add_argument("--vfeat_type", type=str, help="video feature type",
                                 choices=["imagenet_hq", "imagenet_hq_pca", "tsn_rgb_hq", "tsn_rgb_hq_pca", "tsn_flow",
                                          "tsn_flow_pca", "det_hq", "det_hq_pca", "det_hq_rm_dup",
                                          "det_hq_20_100", "det_hq_20_100_pca"])
        self.parser.add_argument("--no_core_driver", action="store_true",
                                 help="hdf5 driver, default use `core` (load into RAM), if specified, use `None`")
        self.parser.add_argument("--no_glove", action="store_true", help="not use glove vectors")
        self.parser.add_argument("--learn_word_embedding", action="store_true", help="fix word embedding")
        self.parser.add_argument("--clip", type=float, default=10., help="perform gradient clip")
        self.parser.add_argument("--resume", type=str, default="", help="path to latest checkpoint")
        self.parser.add_argument("--non_visual", type=int, default=0,
                                 help="add additional vectors for non_visual words")
        self.parser.add_argument("--add_non_visual", action="store_true",
                                 help="count non_visual vectors in when doing weighted sum of the regional vectors")
        self.parser.add_argument("--use_sup_att", action="store_true", help="supervised att, used with use_noun_mask")
        self.parser.add_argument("--att_weight", type=float, default=0.1, help="weight to att loss")
        self.parser.add_argument("--att_iou_thd", type=float, default=0.5, help="IoU threshold for att label")
        self.parser.add_argument("--margin", type=float, default=0.1, help="margin for ranking loss")
        self.parser.add_argument("--num_region", type=int, default=25, help="max number of regions for each image")
        self.parser.add_argument("--att_loss_type", type=str, default="lse", choices=["hinge", "lse"],
                                 help="att loss type, can be hinge loss or its smooth approximation LogSumExp")
        self.parser.add_argument("--scale", type=float, default=10.,
                                 help="multiplier to be applied to similarity score")
        self.parser.add_argument("--alpha", type=float, default=20.,
                                 help="log1p(1 + exp(m + alpha * x)), "
                                      "a high value penalize more when x > 0, less otherwise")
        self.parser.add_argument("--num_hard", type=int, default=2,
                                 help="number of hard negatives, num_hard<=num_negatives")
        self.parser.add_argument("--num_negatives", type=int, default=2,
                                 help="max number of negatives in ranking loss")

        self.parser.add_argument("--hard_negative_start", type=int, default=100,
                                 help="use hard negative when num epochs > hard_negative_start, "
                                      "set to a very high number to stop using it, e.g. 100")
        self.parser.add_argument("--negative_pool_size", type=int, default=0,
                                 help="sample from a pool of hard negative samples (with high scores), "
                                      "instead of a topk hard ones. "
                                      "directly sample topk when negative_pool_size <= num_negatives")
        self.parser.add_argument("--drop_topk", type=int, default=0,
                                 help="do not use the topk negatives")

        # length limit
        self.parser.add_argument("--max_sub_l", type=int, default=50,
                                 help="maxmimum length of all sub sentence 97.71 under 50 for 3 sentences")
        self.parser.add_argument("--max_vid_l", type=int, default=300,
                                 help="maxmimum length of all video sequence")
        self.parser.add_argument("--max_vcpt_l", type=int, default=300,
                                 help="maxmimum length of video seq, 94.25% under 20")
        self.parser.add_argument("--max_q_l", type=int, default=20,
                                 help="maxmimum length of question, 93.91% under 20")  # 25
        self.parser.add_argument("--max_a_l", type=int, default=15,
                                 help="maxmimum length of answer, 98.20% under 15")
        self.parser.add_argument("--max_qa_l", type=int, default=40,
                                 help="maxmimum length of answer, 99.7% <= 40")

        # model config
        self.parser.add_argument("--embedding_size", type=int, default=768, help="word embedding dim")
        self.parser.add_argument("--hsz", type=int, default=128, help="hidden size.")
        self.parser.add_argument("--vocab_size", type=int, default=0, help="vocabulary size")
        self.parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")

        self.parser.add_argument("--input_encoder_n_blocks", type=int, default=1)
        self.parser.add_argument("--input_encoder_n_conv", type=int, default=2)
        self.parser.add_argument("--input_encoder_kernel_size", type=int, default=7)
        self.parser.add_argument("--input_encoder_n_heads", type=int, default=0,
                                 help="number of self-attention heads, 0: do not use it")

        self.parser.add_argument("--cls_encoder_n_blocks", type=int, default=1)
        self.parser.add_argument("--cls_encoder_n_conv", type=int, default=2)
        self.parser.add_argument("--cls_encoder_kernel_size", type=int, default=5)
        self.parser.add_argument("--cls_encoder_n_heads", type=int, default=0,
                                 help="number of self-attention heads, 0: do not use it")

        # paths
        self.parser.add_argument("--glove_path", type=str, default="data/glove.6B.300d.txt",
                                 help="path to download Glove embeddings, you can download it by using "
                                      "`wget http://nlp.stanford.edu/data/glove.6B.zip -q --show-progress`")
        self.parser.add_argument("--word2idx_path", type=str)
        self.parser.add_argument("--eval_object_vocab_path", type=str)
        self.parser.add_argument("--qa_bert_path", type=str, default="")
        self.parser.add_argument("--sub_bert_path", type=str, default="")
        self.parser.add_argument("--train_path", type=str)
        self.parser.add_argument("--valid_path", type=str)
        self.parser.add_argument("--test_path", type=str)
        self.parser.add_argument("--vcpt_path", type=str, default="")
        self.parser.add_argument("--vfeat_path", type=str, default="")
        self.parser.add_argument("--vfeat_size", type=int, default=300, help="dimension of the video feature")
        self.parser.add_argument("--sub_path", type=str, default="")
        self.parser.add_argument("--frm_cnt_path", type=str, default="")

    def display_save(self):
        args = vars(self.opt)
        # Display settings
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        print("[#GPUs]  Using %d / Available %d " % (len(self.opt.device_ids), torch.cuda.device_count()))

        # Save settings
        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(self.opt.results_dir, 'opt.json')  # not yaml file indeed
            save_json_pretty(args, option_file_path)

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        if opt.input_streams is None:
            if isinstance(self, TestOptions):
                opt.input_streams = []
            else:
                raise ValueError("input_streams must be set")

        if opt.debug:
            opt.results_dir_base = opt.results_dir_base.split("/")[0] + "/debug_results"
            opt.no_core_driver = True
            opt.num_workers = 0
        opt.results_dir = opt.results_dir_base + time.strftime("_%Y_%m_%d_%H_%M_%S")

        self.opt = opt

        if isinstance(self, TestOptions):
            options = load_json(os.path.join("results", opt.model_dir, "opt.json"))
            for arg in options:
                if arg not in ["debug"]:
                    setattr(opt, arg, options[arg])
            opt.no_core_driver = True
        else:
            mkdirp(opt.results_dir)
            # save a copy of current code
            code_dir = os.path.dirname(os.path.realpath(__file__))
            code_zip_filename = os.path.join(opt.results_dir, "code.zip")
            make_zipfile(code_dir, code_zip_filename,
                         enclosing_dir="code", exclude_paths=["results"], exclude_extensions=[".pyc", ".ipynb"])
        self.display_save()

        assert opt.num_hard <= opt.num_negatives
        opt.device = torch.device("cuda:%d" % opt.device_ids[0] if opt.device >= 0 else "cpu")
        if opt.device.type == "cuda":
            opt.bsz = opt.bsz * len(opt.device_ids)
            opt.test_bsz = opt.test_bsz * len(opt.device_ids)
        opt.h5driver = None if opt.no_core_driver else "core"
        opt.vfeat_flag = "vfeat" in opt.input_streams
        opt.vcpt_flag = "vcpt" in opt.input_streams
        opt.sub_flag = "sub" in opt.input_streams
        self.opt = opt
        return opt


class TestOptions(BaseOptions):
    """add additional options for evaluating"""
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--model_dir", type=str, help="dir contains the model file")
        self.parser.add_argument("--mode", type=str, default="valid", help="valid/test")
        self.parser.add_argument("--no_strict", action="store_true", help="turn off strict mode in load_state_dict")
