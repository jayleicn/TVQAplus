from __future__ import absolute_import, division, print_function

import os
import sys
import h5py
import pickle
import numpy as np
import torch
import copy
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from utils import load_pickle, load_json, files_exist, get_all_img_ids, computeIoU, \
    flat_list_of_lists, match_stanford_tokenizer, load_glove, get_elements_variable_length, dissect_by_lengths


def filter_list_dicts(list_dicts, key, values):
    """ filter out the dicts with values for key"""
    return [e for e in list_dicts if e[key] in values]


def rm_empty_by_copy(list_array):
    """copy the last non-empty element to replace the empty ones"""
    for idx in range(len(list_array)):
        if len(list_array[idx]) == 0:
            list_array[idx] = list_array[idx-1]
    return list_array


class TVQADataset(Dataset):
    def __init__(self, opt, mode="train"):
        self.opt = opt
        self.is_eval = mode != "train"  # are we running from eval mode
        self.raw_train = load_json(opt.train_path)
        # self.raw_test = load_json(opt.test_path)
        self.raw_valid = load_json(opt.valid_path)
        self.sub_data = load_json(opt.sub_path)
        self.sub_flag = "sub" in opt.input_streams
        self.vfeat_flag = "vfeat" in opt.input_streams
        self.vfeat_type = opt.vfeat_type
        self.qa_bert_h5 = h5py.File(opt.qa_bert_path, "r", driver=opt.h5driver)  # qid + key
        if self.sub_flag:
            self.sub_bert_h5 = h5py.File(opt.sub_bert_path, "r", driver=opt.h5driver)  # vid_name
        if self.vfeat_flag:
            self.vid_h5 = h5py.File(opt.vfeat_path, "r", driver=opt.h5driver)  # add core
        self.vcpt_flag = "vcpt" in opt.input_streams or self.vfeat_flag  # if vfeat, must vcpt
        if self.vcpt_flag:
            self.vcpt_dict = load_pickle(opt.vcpt_path) if opt.vcpt_path.endswith(".pickle") \
                else load_json(opt.vcpt_path)
            if opt.debug:
                self.raw_train = filter_list_dicts(self.raw_train, "vid_name", self.vcpt_dict.keys())
                self.raw_valid = filter_list_dicts(self.raw_valid, "vid_name", self.vcpt_dict.keys())
                # self.raw_test = filter_list_dicts(self.raw_test, "vid_name", self.vcpt_dict.keys())
                print("number of training/valid", len(self.raw_train), len(self.raw_valid))
        self.glove_embedding_path = opt.glove_path
        self.mode = mode
        self.num_region = opt.num_region
        self.use_sup_att = opt.use_sup_att
        self.att_iou_thd = opt.att_iou_thd
        self.cur_data_dict = self.get_cur_dict()

        # tmp
        self.frm_cnt_path = opt.frm_cnt_path
        self.frm_cnt_dict = load_json(self.frm_cnt_path)

        # build/load vocabulary
        self.word2idx_path = opt.word2idx_path
        self.embedding_dim = 300
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<eos>"}
        self.offset = len(self.word2idx)
        text_keys = ["a0", "a1", "a2", "a3", "a4", "q", "sub_text"]
        if not files_exist([self.word2idx_path]):
            print("\nNo cache founded.")
            self.build_word_vocabulary(text_keys, word_count_threshold=2)
        else:
            print("\nLoading cache ...")
            # self.word2idx = load_pickle(self.word2idx_path)
            self.word2idx = load_json(self.word2idx_path)
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def set_mode(self, mode):
        self.mode = mode
        self.is_eval = mode != "train"
        self.cur_data_dict = self.get_cur_dict()

    def get_cur_dict(self):
        if self.mode == 'train':
            return self.raw_train
        elif self.mode == 'valid':
            return self.raw_valid
        elif self.mode == 'test':
            raise NotImplementedError
            # return self.raw_test

    def __len__(self):
        return len(self.cur_data_dict)

    def __getitem__(self, index):
        # 0.5 fps mode
        items = edict()
        items["vid_name"] = self.cur_data_dict[index]["vid_name"]
        vid_name = items["vid_name"]
        items["qid"] = self.cur_data_dict[index]["qid"]
        qid = items["qid"]  # int
        frm_cnt = self.frm_cnt_dict[vid_name]
        located_img_ids = sorted([int(e) for e in self.cur_data_dict[index]["bbox"].keys()])
        start_img_id, end_img_id = located_img_ids[0], located_img_ids[-1]
        indices, start_idx, end_idx = get_all_img_ids(start_img_id, end_img_id, frm_cnt, frame_interval=6)
        items["anno_st_idx"] = start_idx
        indices = np.array(indices) - 1  # since the frame (image) index from 1

        items["ts_label"] = self.get_ts_label(self.cur_data_dict[index]["ts"][0],
                                              self.cur_data_dict[index]["ts"][1],
                                              frm_cnt,
                                              indices,
                                              fps=3)
        items["ts"] = self.cur_data_dict[index]["ts"]  # [st (float), ed (float)]
        items["image_indices"] = (indices + 1).tolist()
        items["image_indices"] = items["image_indices"]

        if self.is_eval and self.vfeat_flag:
            # add boxes
            boxes = self.vcpt_dict[vid_name]["boxes"]  # full resolution
            lowered_boxes = [boxes[idx][:self.num_region] for idx in indices]
            items["boxes"] = lowered_boxes[start_idx:end_idx+1]
        else:
            items["boxes"] = None

        # add correct answer
        ca_idx = int(self.cur_data_dict[index]["answer_idx"])
        items["target"] = ca_idx

        # add q-answers
        answer_keys = ["a0", "a1", "a2", "a3", "a4"]
        qa_sentences = [self.numericalize(self.cur_data_dict[index]["q"]
                        + " " + self.cur_data_dict[index][k], eos=False) for k in answer_keys]
        qa_sentences_bert = [torch.from_numpy(
            np.concatenate([self.qa_bert_h5[str(qid) + "_q"], self.qa_bert_h5[str(qid) + "_" + k]], axis=0))
            for k in answer_keys]
        q_l = self.cur_data_dict[index]["q_len"]
        ca_l = self.cur_data_dict[index]["a{}_len".format(ca_idx)]
        items["q_l"] = q_l
        items["qas"] = qa_sentences
        items["qas_bert"] = qa_sentences_bert

        # add sub
        if self.sub_flag:
            img_aligned_sub_indices, raw_sub_n_tokens = self.get_aligned_sub_indices(
                indices + 1,
                self.sub_data[vid_name]["sub_text"],
                self.sub_data[vid_name]["sub_time"],
                mode="nearest")
            try:
                sub_bert_embed = dissect_by_lengths(self.sub_bert_h5[vid_name][:], raw_sub_n_tokens, dim=0)
            except AssertionError as e:  # 35 QAs from 7 videos
                sub_bert_embed = dissect_by_lengths(self.sub_bert_h5[vid_name][:], raw_sub_n_tokens,
                                                    dim=0, assert_equal=False)
                sub_bert_embed = rm_empty_by_copy(sub_bert_embed)
            assert len(sub_bert_embed) == len(raw_sub_n_tokens)  # we did not truncate when extract embeddings

            items["sub_bert"] = [torch.from_numpy(np.concatenate([sub_bert_embed[in_idx] for in_idx in e], axis=0))
                                 for e in img_aligned_sub_indices]
            aligned_sub_text = self.get_aligned_sub(self.sub_data[vid_name]["sub_text"],
                                                    img_aligned_sub_indices)
            items["sub"] = [self.numericalize(e, eos=False) for e in aligned_sub_text]
        else:
            items["sub_bert"] = [torch.zeros(2, 2)] * 2
            items["sub"] = [torch.zeros(2, 2)] * 2

        if self.vfeat_flag or self.vcpt_flag:
            region_counts = self.vcpt_dict[vid_name]["counts"]  # full resolution
            localized_lowered_region_counts = \
                [min(region_counts[idx], self.num_region) for idx in indices][start_idx:end_idx+1]

        # add vcpt
        if self.vcpt_flag:
            lower_res_obj_labels = get_elements_variable_length(
                self.vcpt_dict[vid_name]["object"], indices, cnt_list=None, max_num_region=self.num_region)
            obj_labels = lower_res_obj_labels
            items["vcpt"] = self.numericalize_hier_vcpt(obj_labels)
            items["object_labels"] = obj_labels
        else:
            items["vcpt"] = [[0, 0], [0, 0]]

        # add visual feature
        if self.vfeat_flag:
            lowered_vfeat = get_elements_variable_length(
                self.vid_h5[vid_name][:], indices, cnt_list=region_counts, max_num_region=self.num_region)
            cur_vfeat = lowered_vfeat

            items["vfeat"] = [torch.from_numpy(e) for e in cur_vfeat]
        else:
            items["vfeat"] = [torch.zeros(2, 2)] * 2

        # add att
        if (self.use_sup_att or self.is_eval) and self.vfeat_flag:  # in order to eval for models without sup_att
            q_ca_sentence = self.cur_data_dict[index]["q"] + " " + \
                            self.cur_data_dict[index]["a{}".format(ca_idx)]
            iou_data = self.get_iou_data(self.cur_data_dict[index]["bbox"], self.vcpt_dict[vid_name], frm_cnt)
            items["att_labels"] = self.mk_att_label(
                iou_data, q_ca_sentence, localized_lowered_region_counts, q_l + ca_l + 1,
                iou_thd=self.att_iou_thd, single_box=self.is_eval)
        else:
            items["att_labels"] = [torch.zeros(2, 2, 2)] * 2
        return items

    @classmethod
    def get_ts_label(cls, st, ed, num_frame, indices, fps=3):
        """ Get temporal supervise signal
        Args:
            st (float):
            ed (float):
            num_frame (int):
            indices (np.ndarray): fps0.5 indices
            fps (int): frame rate used to extract the frames
        Returns:
            sup_ts_type==`st_ed`: [start_idx, end_idx]
        """
        max_num_frame = 300.
        if num_frame > max_num_frame:
            st, ed = [(max_num_frame / num_frame) * fps * ele for ele in [st, ed]]
        else:
            st, ed = [fps * ele for ele in [st, ed]]

        start_idx = np.searchsorted(indices, st, side="left")
        end_idx = np.searchsorted(indices, ed, side="right")
        max_len = len(indices)
        if not start_idx < max_len:
            start_idx -= 1
        if not end_idx < max_len:
            end_idx -= 1
        if start_idx == end_idx:
            st_ed = [start_idx, end_idx]
        else:
            st_ed = [start_idx, end_idx-1]  # this is the correct formula

        return st_ed  # (2, )

    @classmethod
    def line_to_words(cls, line, eos=True, downcase=True):
        eos_word = "<eos>"
        words = line.lower().split() if downcase else line.split()
        # !!!! remove comma here, since they are too many of them, !!! no removing  # TODO
        # words = [w for w in words if w != ","]
        words = [w for w in words]
        words = words + [eos_word] if eos else words
        return words

    @classmethod
    def find_match(cls, subtime, value, mode="larger", span=1.5):
        """closet value in an array to a given value"""
        if mode == "nearest":  # closet N samples
            return sorted((np.abs(subtime - value)).argsort()[:2].tolist())
        elif mode == "span":  # with a specified time span
            return_indices = np.nonzero(np.abs(subtime - value) < span)[0].tolist()
            if value <= 2:
                return_indices = np.nonzero(subtime - 2 <= 0)[0].tolist() + return_indices
            return return_indices
        elif mode == "larger":
            idx = max(0, np.searchsorted(subtime, value, side="left") - 1)
            return_indices = [idx - 1, idx, idx + 1]
            return_indices = [idx for idx in return_indices if 0 <= idx < len(subtime)]
            return return_indices

    @classmethod
    def get_aligned_sub_indices(cls, img_ids, subtext, subtime, fps=3, mode="larger"):
        """ Get aligned subtitle for each frame, for each frame, use the two subtitle
        sentences that are most close to it
        Args:
            img_ids (list of int): image file ids, note the image index starts from 1. Is one possible???
            subtext (str): tokenized subtitle sentences concatenated by "<eos>".
            subtime (list of float): a list of timestamps from the subtile file, each marks the start
                of each subtile sentence. It should have the same length as the "<eos>" splitted subtext.
            fps (int): frame per second when extracting the video
            mode (str): nearest or larger
        Returns:
            a list of str, each str should be aligned with an image indicated by img_ids.
        """
        subtext = subtext.split(" <eos> ")  # note the spaces
        raw_sub_n_tokens = [len(s.split()) for s in subtext]
        assert len(subtime) == len(subtext)
        img_timestamps = np.array(img_ids) / fps  # roughly get the timestamp for the
        img_aligned_sentence_indices = []  # list(list)
        for t in img_timestamps:
            img_aligned_sentence_indices.append(cls.find_match(subtime, t, mode=mode))
        return img_aligned_sentence_indices, raw_sub_n_tokens

    @classmethod
    def get_aligned_sub(cls, subtext, img_aligned_sentence_indices):
        subtext = subtext.split(" <eos> ")  # note the spaces
        return [" ".join([subtext[inner_idx] for inner_idx in e]) for e in img_aligned_sentence_indices]

    def mk_noun_mask(self, noun_indices_q, noun_indices_a, q_l, a_l, eos=True):
        """ mask is a ndarray (num_q_words + num_ca_words + 1, )
        removed nouns that are not in the vocabulary
        Args:
            noun_indices_q (list): each element is [index, word]
            noun_indices_a (list):
            q_l (int):
            a_l (int):
            eos

        Returns:

        """
        noun_indices_q = [e[0] for e in noun_indices_q if e[1].lower() in self.word2idx]
        noun_indices_a = [e[0] + q_l for e in noun_indices_a if e[1].lower() in self.word2idx]
        noun_indices = np.array(noun_indices_q + noun_indices_a) - 1
        mask = np.zeros(q_l + a_l + 1) if eos else np.zeros(q_l + a_l)
        if len(noun_indices) != 0:  # seems only 1 instance has no indices
            mask[noun_indices] = 1
        return mask

    @classmethod
    def get_labels_single_box(cls, single_box, detected_boxes):
        """return a list of IoUs"""
        gt_box = [single_box["left"], single_box["top"],
                  single_box["left"] + single_box["width"],
                  single_box["top"] + single_box["height"]]  # [left, top, right, bottom]
        IoUs = [float("{:.4f}".format(computeIoU(gt_box, d_box))) for d_box in detected_boxes]
        return IoUs

    def get_iou_data(self, gt_box_data_i, meta_data_i, frm_cnt_i):
        """
        meta_data (dict):  with vid_name as key,
        add iou_data entry, organized similar to bbox_data
        """
        frm_cnt_i = frm_cnt_i + 1  # add extra 1 since img_ids are 1-indexed
        iou_data_i = {}
        img_ids = sorted(gt_box_data_i.keys(), key=lambda x: int(x))
        img_ids = [e for e in img_ids if int(e) < frm_cnt_i]
        for img_id in img_ids:
            iou_data_i[img_id] = []
            cur_detected_boxes = meta_data_i["boxes"][int(img_id) - 1]
            for box in gt_box_data_i[img_id]:
                iou_list = self.get_labels_single_box(box, cur_detected_boxes)
                iou_data_i[img_id].append({
                    "iou": iou_list,
                    "label": box["label"],
                    "img_id": img_id
                })
        return iou_data_i

    @classmethod
    def mk_att_label(cls, iou_data, q_ca_sentence, region_cnts, ca_len, iou_thd=0.5, single_box=False):
        """return a list(dicts) of length num_imgs, each dict with word indices as keys,
        with corresponding region index as values.
        iou_data:
        q_ca_sentence: q(str) + " " + ca(str)
        region_cnts: list(int)
        ca_len: int, number of words for the concatenation of question the correct answer, +1 for eos
        single_box (bool): return a single object box for each gt box (the one with highest IoU)
        """
        img_ids = sorted(iou_data.keys(), key=lambda x: int(x))
        q_ca_words = q_ca_sentence.split()
        att_label = [np.zeros((ca_len, cnt)) for cnt in region_cnts]  # #imgs * (#words, #regions)
        for idx, img_id in enumerate(img_ids):  # within a single image
            cur_img_iou_info = iou_data[img_id]
            cur_labels = [e["label"] for e in cur_img_iou_info]  # might be upper case
            for noun_idx in range(ca_len-1):  # do not count <EOS> in
                # find the gt boxes (possibly > 1) under the same label
                cur_noun = q_ca_words[noun_idx]
                cur_box_indices = [box_idx for box_idx, label in enumerate(cur_labels)
                                   if label.lower() == cur_noun.lower()]

                # find object boxes that has high IoU with gt boxes, 1 or more for each gt box (single_box)
                cur_iou_mask = None
                for box_idx in cur_box_indices:
                    if cur_iou_mask is None:
                        # why is [:region_cnts[idx]] The cnt here is actually after min(cnt, max_num_regions)
                        if single_box:
                            cur_ios_mask_len = len(cur_img_iou_info[box_idx]["iou"][:region_cnts[idx]])
                            cur_iou_mask = np.zeros(cur_ios_mask_len)
                            if np.max(cur_img_iou_info[box_idx]["iou"][:region_cnts[idx]]) >= iou_thd:
                                cur_iou_mask[np.argmax(cur_img_iou_info[box_idx]["iou"][:region_cnts[idx]])] = 1
                        else:
                            cur_iou_mask = np.array(cur_img_iou_info[box_idx]["iou"][:region_cnts[idx]]) >= iou_thd
                    else:
                        if single_box:  # assume the high IoU boxes for the same label will not be the same
                            if np.max(cur_img_iou_info[box_idx]["iou"][:region_cnts[idx]]) >= iou_thd:
                                cur_iou_mask[np.argmax(cur_img_iou_info[box_idx]["iou"][:region_cnts[idx]])] = 1
                        else:
                            # [True, False] + [True, True] = [True, True]
                            cur_iou_mask += np.array(cur_img_iou_info[box_idx]["iou"][:region_cnts[idx]]) >= iou_thd
                if cur_iou_mask is not None:
                    # less than num_regions is possible,
                    # we assume the attention is evenly paid to overlapped boxes
                    if cur_iou_mask.sum() != 0:
                        cur_iou_mask = cur_iou_mask.astype(np.float32) / cur_iou_mask.sum()  # TODO
                    att_label[idx][noun_idx] = cur_iou_mask
        return [torch.from_numpy(e) for e in att_label]  # , att_label_mask

    def numericalize(self, sentence, eos=True, match=False):
        """convert words to indices, match stanford tokenizer"""
        if match:
            sentence = match_stanford_tokenizer(sentence)
        sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
                            for w in self.line_to_words(sentence, eos=eos)]  # 1 is <unk>, unknown
        return sentence_indices

    def numericalize_hier_vcpt(self, vcpt_words_list):
        """vcpt_words_list is a list of sublist, each sublist contains words"""
        sentence_indices = []
        for i in range(len(vcpt_words_list)):
            # some labels are 'tennis court', keep the later word
            words = [e.split()[-1] for e in vcpt_words_list[i]]
            sentence_indices.append([self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
                                     for w in words])
        return sentence_indices

    def numericalize_vcpt(self, vcpt_sentence):
        """convert words to indices, additionally removes duplicated attr-object pairs"""
        attr_obj_pairs = vcpt_sentence.lower().split(",")  # comma is also removed
        attr_obj_pairs = [e.strip() for e in attr_obj_pairs]
        unique_pairs = []
        for pair in attr_obj_pairs:
            if pair not in unique_pairs:
                unique_pairs.append(pair)
        words = []
        for pair in unique_pairs:
            words.extend(pair.split())
        words.append("<eos>")
        sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
                            for w in words]
        return sentence_indices

    def build_word_vocabulary(self, text_keys, word_count_threshold=0):
        """
        borrowed this implementation from @karpathy's neuraltalk.
        """
        print("Building word vocabulary starts.\n")
        all_sentences = []
        for k in text_keys:
            all_sentences.extend(self.raw_train[k])

        word_counts = {}
        for sentence in all_sentences:
            for w in self.line_to_words(sentence, eos=False, downcase=True):
                word_counts[w] = word_counts.get(w, 0) + 1

        # vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold and w not in self.word2idx.keys()]
        print("Vocabulary Size %d (<pad> <unk> <eos> excluded) using word_count_threshold %d.\n" %
              (len(vocab), word_count_threshold))

        # build index and vocabularies
        for idx, w in enumerate(vocab):
            self.word2idx[w] = idx + self.offset
            self.idx2word[idx + self.offset] = w

        print("word2idx size: %d, idx2word size: %d.\n" % (len(self.word2idx), len(self.idx2word)))
        # Make glove embedding.
        print("Loading glove embedding at path : %s.\n" % self.glove_embedding_path)
        glove_full = load_glove(self.glove_embedding_path)
        print("Glove Loaded, building word2idx, idx2word mapping.\n")

        glove_matrix = np.zeros([len(self.idx2word), self.embedding_dim])
        glove_keys = glove_full.keys()
        for i in tqdm(range(len(self.idx2word))):
            w = self.idx2word[i]
            w_embed = glove_full[w] if w in glove_keys else np.random.randn(self.embedding_dim) * 0.4
            glove_matrix[i, :] = w_embed
        self.vocab_embedding = glove_matrix
        print("vocab embedding size is :", glove_matrix.shape)

        print("Saving cache files at ./cache.\n")
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        pickle.dump(self.word2idx, open(self.word2idx_path, 'w'))
        pickle.dump(self.idx2word, open(self.idx2word_path, 'w'))
        pickle.dump(glove_matrix, open(self.vocab_embedding_path, 'w'))

        print("Building  vocabulary done.\n")


def pad_sequence_3d_label(sequences, sequences_masks):
    """
    Args:
        sequences: list(3d torch.Tensor)
        sequences_masks: list(torch.Tensor) of the same shape as sequences,
            individual mask is the result of masking the individual element

    Returns:

    """
    shapes = [seq.shape for seq in sequences]
    lengths_1 = [s[0] for s in shapes]
    lengths_2 = [s[1] for s in shapes]
    lengths_3 = [s[2] for s in shapes]
    padded_seqs = torch.zeros(len(sequences), max(lengths_1), max(lengths_2), max(lengths_3)).float()
    mask = copy.deepcopy(padded_seqs)
    for idx, seq in enumerate(sequences):
        padded_seqs[idx, :lengths_1[idx], :lengths_2[idx], :lengths_3[idx]] = seq
        mask[idx, :lengths_1[idx], :lengths_2[idx], :lengths_3[idx]] = sequences_masks[idx]
    return padded_seqs, mask


def pad_sequences_2d(sequences, dtype=torch.long):
    """ Pad a double-nested list or a sequence of n-d torch tensor into a (n+1)-d tensor,
        only allow the first two dims has variable lengths
    Args:
        sequences: list(n-d tensor or list)
        dtype: torch.long for word indices / torch.float (float32) for other cases

    Returns:

    Examples:
        >>> test_data_list = [[[1, 3, 5], [3, 7, 4, 1]], [[98, 34, 11, 89, 90], [22], [34, 56]],]
        >>> pad_sequences_2d(test_data_list, dtype=torch.long)  # torch.Size([2, 3, 5])
        >>> test_data_3d = [torch.randn(2,2,4), torch.randn(4,3,4), torch.randn(1,5,4)]
        >>> pad_sequences_2d(test_data_3d, dtype=torch.float)  # torch.Size([2, 3, 5])
        >>> test_data_3d2 = [[torch.randn(2,4), ], [torch.randn(3,4), torch.randn(5,4)]]
        >>> pad_sequences_2d(test_data_3d2, dtype=torch.float)  # torch.Size([2, 3, 5])
    """
    bsz = len(sequences)
    para_lengths = [len(seq) for seq in sequences]
    max_para_len = max(para_lengths)
    sen_lengths = [[len(word_seq) for word_seq in seq] for seq in sequences]
    max_sen_len = max(flat_list_of_lists(sen_lengths))

    if isinstance(sequences[0], torch.Tensor):
        extra_dims = sequences[0].shape[2:]
    elif isinstance(sequences[0][0], torch.Tensor):
        extra_dims = sequences[0][0].shape[1:]
    else:
        sequences = [[torch.LongTensor(word_seq) for word_seq in seq] for seq in sequences]
        extra_dims = ()

    padded_seqs = torch.zeros((bsz, max_para_len, max_sen_len) + extra_dims, dtype=dtype)
    mask = torch.zeros(bsz, max_para_len, max_sen_len).float()

    for b_i in range(bsz):
        for sen_i, sen_l in enumerate(sen_lengths[b_i]):
            padded_seqs[b_i, sen_i, :sen_l] = sequences[b_i][sen_i]
            mask[b_i, sen_i, :sen_l] = 1
    return padded_seqs, mask  # , sen_lengths


def pad_sequences_1d(sequences, dtype=torch.long):
    """ Pad a single-nested list or a sequence of n-d torch tensor into a (n+1)-d tensor,
        only allow the first dim has variable lengths
    Args:
        sequences: list(n-d tensor or list)
        dtype: torch.long for word indices / torch.float (float32) for other cases
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
    """
    if isinstance(sequences[0], list):
        sequences = [torch.tensor(s, dtype=dtype) for s in sequences]
    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    padded_seqs = torch.zeros((len(sequences), max(lengths)) + extra_dims, dtype=dtype)
    mask = torch.zeros(len(sequences), max(lengths)).float()
    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths


def make_mask_from_length(lengths):
    mask = torch.zeros(len(lengths), max(lengths)).float()
    for idx, l in enumerate(lengths):
        mask[idx, :l] = 1
    return mask


def pad_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    """
    # separate source and target sequences
    batch = edict()
    batch["qas"], batch["qas_mask"] = pad_sequences_2d([d["qas"] for d in data], dtype=torch.long)
    batch["qas_bert"], _ = pad_sequences_2d([d["qas_bert"] for d in data], dtype=torch.float)
    batch["sub"], batch["sub_mask"] = pad_sequences_2d([d["sub"] for d in data], dtype=torch.long)
    batch["sub_bert"], _ = pad_sequences_2d([d["sub_bert"] for d in data], dtype=torch.float)
    batch["vid_name"] = [d["vid_name"] for d in data]
    batch["qid"] = [d["qid"] for d in data]
    batch["target"] = torch.tensor([d["target"] for d in data], dtype=torch.long)
    batch["vcpt"], batch["vcpt_mask"] = pad_sequences_2d([d["vcpt"] for d in data], dtype=torch.long)
    batch["vid"], batch["vid_mask"] = pad_sequences_2d([d["vfeat"] for d in data], dtype=torch.float)
    # no need to pad these two, since we will break down to instances anyway
    batch["att_labels"] = [d["att_labels"] for d in data]  # a list, each will be (num_img, num_words)
    batch["anno_st_idx"] = [d["anno_st_idx"] for d in data]  # list(int)
    if data[0]["ts_label"] is None:
        batch["ts_label"] = None
    elif isinstance(data[0]["ts_label"], list):  # (st_ed, ce)
        batch["ts_label"] = dict(
            st=torch.LongTensor([d["ts_label"][0] for d in data]),
            ed=torch.LongTensor([d["ts_label"][1] for d in data]),
        )
        batch["ts_label_mask"] = make_mask_from_length([len(d["image_indices"]) for d in data])
    elif isinstance(data[0]["ts_label"], torch.Tensor):  # (st_ed, bce) or frm
        batch["ts_label"], batch["ts_label_mask"] = pad_sequences_1d([d["ts_label"] for d in data], dtype=torch.float)
    else:
        raise NotImplementedError

    batch["ts"] = [d["ts"] for d in data]
    batch["image_indices"] = [d["image_indices"] for d in data]
    batch["q_l"] = [d["q_l"] for d in data]

    batch["boxes"] = [d["boxes"] for d in data]
    batch["object_labels"] = [d["object_labels"] for d in data]
    return batch


def prepare_inputs(batch, max_len_dict=None, device="cuda"):
    """clip and move input data to gpu"""
    model_in_dict = edict()

    # qas (B, 5, #words, D)
    max_qa_l = min(batch["qas"].shape[2], max_len_dict["max_qa_l"])
    model_in_dict["qas"] = batch["qas"][:, :, :max_qa_l].to(device)
    model_in_dict["qas_bert"] = batch["qas_bert"][:, :, :max_qa_l].to(device)
    model_in_dict["qas_mask"] = batch["qas_mask"][:, :, :max_qa_l].to(device)

    # (B, #imgs, #words, D)
    model_in_dict["sub"] = batch["sub"][:, :max_len_dict["max_vid_l"], :max_len_dict["max_sub_l"]].to(device)
    model_in_dict["sub_bert"] = batch["sub_bert"][:, :max_len_dict["max_vid_l"], :max_len_dict["max_sub_l"]].to(device)
    model_in_dict["sub_mask"] = batch["sub_mask"][:, :max_len_dict["max_vid_l"], :max_len_dict["max_sub_l"]].to(device)

    # context, vid (B, #imgs, #regions, D), vcpt (B, #imgs, #regions)
    ctx_keys = ["vid", "vcpt"]
    for k in ctx_keys:
        max_l = min(batch[k].shape[1], max_len_dict["max_{}_l".format(k)])
        model_in_dict[k] = batch[k][:, :max_l].to(device)
        mask_key = "{}_mask".format(k)
        model_in_dict[mask_key] = batch[mask_key][:, :max_l].to(device)

    # att_label (B, #imgs, #qa_words, #regions)
    max_att_imgs = min(max([len(d) for d in batch["att_labels"]]), max_len_dict["max_vid_l"])
    max_att_words = min(max([d[0].shape[0] for d in batch["att_labels"]]), max_len_dict["max_qa_l"])
    model_in_dict["att_labels"] = [[inner_d[:max_att_words].to(device) for inner_d in d[:max_att_imgs]]
                                   for d in batch["att_labels"]]
    model_in_dict["anno_st_idx"] = batch["anno_st_idx"]

    if batch["ts_label"] is None:
        model_in_dict["ts_label"] = None
        model_in_dict["ts_label_mask"] = None
    elif isinstance(batch["ts_label"], dict):  # (st_ed, ce)
        model_in_dict["ts_label"] = dict(
            st=batch["ts_label"]["st"].to(device),
            ed=batch["ts_label"]["ed"].to(device),
        )
        model_in_dict["ts_label_mask"] = batch["ts_label_mask"][:, :max_len_dict["max_vid_l"]].to(device)
    else:  # frm-wise or (st_ed, bce)
        model_in_dict["ts_label"] = batch["ts_label"][:, :max_len_dict["max_vid_l"]].to(device)
        model_in_dict["ts_label_mask"] = batch["ts_label_mask"][:, :max_len_dict["max_vid_l"]].to(device)

    # target
    model_in_dict["target"] = batch["target"].to(device)

    # others
    model_in_dict["qid"] = batch["qid"]
    model_in_dict["vid_name"] = batch["vid_name"]

    targets = model_in_dict["target"]
    qids = model_in_dict["qid"]
    model_in_dict["ts"] = batch["ts"]
    model_in_dict["q_l"] = batch["q_l"]
    model_in_dict["image_indices"] = batch["image_indices"]
    model_in_dict["boxes"] = batch["boxes"]
    model_in_dict["object_labels"] = batch["object_labels"]
    return model_in_dict, targets, qids


def find_match(subtime, time_array, mode="larger"):
    """find closet value in an array to a given value
    subtime (float):
    time_array (np.ndarray): (N, )
    """
    if mode == "nearest":
        return (np.abs(subtime - time_array)).argsort()[:2].tolist()
    elif mode == "larger":
        idx = max(0, np.searchsorted(subtime, time_array, side="left") - 1)
        return_indices = [idx-1, idx, idx+1]
        # return_indices = [idx, idx+1]
        return_indices = [idx for idx in return_indices if 0 <= idx < len(subtime)]
        return return_indices
    else:
        raise NotImplementedError
