__author__ = "Jie Lei"

import re
import os
import json
import math
import numpy as np
import zipfile
import h5py

from tqdm import tqdm
try:
    import cPickle as pickle
except:
    import pickle


def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)


def save_json_pretty(data, file_path):
    """save formatted json, use this one for some json config files"""
    with open(file_path, "w") as f:
        f.write(json.dumps(data, indent=4, sort_keys=True))


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_pickle(data, data_path, highest=False):
    protocol = 2 if highest else 0
    with open(data_path, "w") as f:
        pickle.dump(data, f, protocol=protocol)


def load_jsonl_as_dict(filepath, target_k):
    """each line is a dict as well"""
    lines = read_json_lines(filepath)
    assert target_k in lines[0]
    dict_obj = {l[target_k]: l for l in lines}
    return dict_obj


def read_json_lines(file_path):
    print("reading data...")
    with open(file_path, "r") as f:
        lines = []
        value_err_cnt = 0
        for l in tqdm(f.readlines()):
            try:
                loaded_l = json.loads(l.strip("\n"))
                lines.append(loaded_l)
            except ValueError as e:
                value_err_cnt += 1
                continue
    return lines


# def load_pickle(file_path):
#     with open(file_path, "r") as f:
#         return pickle.load(f)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def files_exist(filepath_list):
    """check whether all the files exist"""
    for ele in filepath_list:
        if not os.path.exists(ele):
            return False
    return True


def load_glove(filename):
    """ returns { word (str) : vector_embedding (torch.FloatTensor) }
    """
    glove = {}
    with open(filename) as f:
        for line in f.readlines():
            values = line.strip("\n").split(" ")  # space separator
            word = values[0]
            vector = np.asarray([float(e) for e in values[1:]])
            glove[word] = vector
    return glove


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def merge_dicts(list_dicts):
    merged_dict = list_dicts[0].copy()
    for i in range(1, len(list_dicts)):
        merged_dict.update(list_dicts[i])
    return merged_dict


def get_parent_dir(file_path):
    return os.path.abspath(os.path.join(file_path, os.pardir))


def dissect_by_lengths(np_array, lengths, dim=0, assert_equal=True):
    """Dissect an array (N, D) into a list a sub-array,
    np_array.shape[0] == sum(lengths), Output is a list of nd arrays, singlton dimention is kept"""
    if assert_equal:
        assert len(np_array) == sum(lengths)
    length_indices = [0, ]
    for i in range(len(lengths)):
        length_indices.append(length_indices[i] + lengths[i])
    if dim == 0:
        array_list = [np_array[length_indices[i]:length_indices[i+1]] for i in range(len(lengths))]
    elif dim == 1:
        array_list = [np_array[:, length_indices[i]:length_indices[i + 1]] for i in range(len(lengths))]
    elif dim == 2:
        array_list = [np_array[:, :, length_indices[i]:length_indices[i + 1]] for i in range(len(lengths))]
    else:
        raise NotImplementedError
    return array_list


def get_all_img_ids(interval_start_img_id, interval_end_img_id, num_imgs, frame_interval=6):
    """ get 0.5fps image ids sequence that contains the localized img_ids
    this should be used for each question in bbt (since I made a stupid mistake T_T), note img_ids are 1-indexed
    :param interval_start_img_id: (int) the first img id used
    :param interval_end_img_id: (int) the last img id used
    :param num_imgs: (int) total number of images for the video
    :param frame_interval: (int)
    :return: indices (list), located_mask (list)
    """
    real_start = interval_start_img_id % frame_interval  # residual
    real_start = frame_interval if real_start == 0 else real_start
    indices = range(real_start, min(num_imgs+1, 301), frame_interval)
    assert 0 not in indices
    mask_start_idx = indices.index(interval_start_img_id)
    # mask_end_idx = indices.index(interval_end_img_id)
    # some indices are larger than num_imgs, TODO should be addressed in data preprocessing part
    if interval_end_img_id in indices:
        mask_end_idx = indices.index(interval_end_img_id)
    else:
        mask_end_idx = len(indices) - 1
    return indices, mask_start_idx, mask_end_idx


def make_large_resolution_indices(indices, resolution=16):
    """
    :param indices: (list) of int
    :param resolution: (int)
    :return:
    """
    indices = np.array(indices) * resolution
    expanded_indices = flat_list_of_lists([range(ele-resolution, ele) for ele in indices])
    return expanded_indices


def get_elements_from_indices(elements, indices, resolution=1):
    """ get selected elements specified by indices
    :param elements: (list) or (numpy.ndarray)
    :param indices: (list)
    :param resolution: (int) how many elements each index refer to, useful for detection features,
                        where each image has multiple features
    :return:
    """
    if resolution == 1:
        if isinstance(elements, list):
            return [elements[idx] for idx in indices]
        elif isinstance(elements, np.ndarray):
            return elements[indices]
        else:
            raise ValueError("[resolution=1] elements must be an instance of (list) or (numpy.ndarray)")
    else:
        if isinstance(elements, np.ndarray):
            indices = make_large_resolution_indices(indices, resolution=resolution)
            return elements[indices]
        else:
            raise ValueError("[resolution>1] elements must be an instance of (numpy.ndarray)")


def get_elements_variable_length(elements, indices_list, cnt_list=None, max_num_region=16, assert_equal=True):
    """
    Args:
        elements: list(list) or numpy.ndarray
        cnt_list: list(int), stores the number of regions for each image
        indices_list: list(int), stores the image indices to use
        max_num_region: (int) only take the top max_num_region
        assert_equal:

    Returns:
        list(ndarray) or list(list)
    """
    if isinstance(elements, np.ndarray):
        elements = dissect_by_lengths(elements, cnt_list, assert_equal=assert_equal)
    elif isinstance(elements, list):
        pass
    else:
        raise NotImplementedError
    return [elements[idx][:max_num_region] for idx in indices_list]


def get_bbox_target_single_box(single_box, spatial_dim=7, img_w=640., img_h=360., thd=0.5):
    """
    :param single_box: a single box
    :param spatial_dim:
    :param img_w:
    :param img_h:
    :param thd: round thd
    :return:
    """
    top = single_box["top"]
    left = single_box["left"]
    bottom = top + single_box["height"]
    right = left + single_box["width"]

    # map to 224x224 to 7x7
    top = int(math.floor((top * spatial_dim) / img_h + thd))
    bottom = int(math.ceil((bottom * spatial_dim) / img_h - thd))
    left = int(math.floor((left * spatial_dim) / img_w + thd))
    right = int(math.ceil((right * spatial_dim) / img_w - thd))
    gt_att_map = np.zeros([spatial_dim, spatial_dim]).astype(np.float32)
    gt_att_map[top: bottom+1, left:right+1] = 1
    # print(top, bottom, left, right)
    return gt_att_map


def get_bbox_target_for_single_img(list_bboxes, spatial_dim=7):
    """get bbox for single image, with 0+ bboxes. Note if no bbox, a all one array will be used"""
    if len(list_bboxes) == 0:
        cur_map = np.ones([spatial_dim, spatial_dim]).astype(np.float32)
    elif len(list_bboxes) == 1:
        cur_map = get_bbox_target_single_box(list_bboxes[0], spatial_dim=spatial_dim)
    else:
        multiple_maps = [get_bbox_target_single_box(ele, spatial_dim=spatial_dim) for ele in list_bboxes]
        cur_map = (sum(multiple_maps) > 0).astype(np.float32)
    return cur_map.reshape(-1)  # 49


def get_bbox_target(bbox_data_dict, num_imgs, spatial_dim=7):
    """
    :param bbox_data_dict:
    :param num_imgs: max_num_imgs
    :param spatial_dim:
    :return:
    """
    # note img_ids are 1-indexed
    # some indices are larger than num_imgs, TODO should be addressed in data preprocessing part
    img_ids = [int(k) for k in bbox_data_dict.keys() if int(k) <= num_imgs]
    img_ids.sort()  # increasing
    mask = np.asarray([int(len(bbox_data_dict[str(k)]) > 0) for k in img_ids])  # mask is 0 if no bbox
    # try:
    bbox_target = np.stack([
        get_bbox_target_for_single_img(bbox_data_dict[str(k)], spatial_dim=spatial_dim) for k in img_ids
    ], axis=0)  # Nx49
    # except ValueError as e:
    #     print(e.message)
    #     print("bbox_data_dict", bbox_data_dict.keys())  # [u'1', u'19', u'13', u'7']
    #     print("num_imgs", num_imgs)  # 182
    return bbox_target, mask  # Nx49, N


def get_dir_size(dir_path, unit="MB"):
    """Get size of a directory, unit can be [B, KB, MB, GB]"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)

    # convert to human readable unit
    if unit == "B":
        pass
    elif unit == "KB":
        total_size /= 1024.
    elif unit == "MB":
        total_size /= (1024. ** 2)
    elif unit == "GB":
        total_size /= (1024. ** 3)
    else:
        raise ValueError("Wrong value for unit, ")
    return total_size


def make_zipfile(src_dir, save_path, enclosing_dir="", exclude_paths=None, exclude_extensions=None):
    """make a zip file of root_dir, save it to save_path.
    exclude_paths will be excluded if it is a subdir of root_dir.
    An enclosing_dir is added is specified.
    """
    abs_src = os.path.abspath(src_dir)
    with zipfile.ZipFile(save_path, "w") as zf:
        for dirname, subdirs, files in os.walk(src_dir):
            # print("dirname", dirname)
            # print("subdirs", subdirs)
            # print("files", files)
            if exclude_paths is not None:
                for e_p in exclude_paths:
                    if e_p in subdirs:
                        subdirs.remove(e_p)
            arcname = os.path.join(enclosing_dir, dirname[len(abs_src) + 1:])
            zf.write(dirname, arcname)
            for filename in files:
                if exclude_extensions is not None:
                    if os.path.splitext(filename)[1] in exclude_extensions:
                        continue  # do not zip it
                absname = os.path.join(dirname, filename)
                arcname = os.path.join(enclosing_dir, absname[len(abs_src) + 1:])
                zf.write(absname, arcname)


def match_stanford_tokenizer(line):
    """To match stanford tokenizer results"""
    line = re.sub("'", " ' ", line)
    line = re.sub("n ' t", "n't", line)
    line = re.sub("' s", "'s", line)
    line = re.sub("' re", "'re", line)
    line = re.sub("' d", "'d", line)
    line = re.sub("' ll", "'ll", line)
    line = re.sub("' m", "'m", line)
    line = re.sub("' ve", "'ve", line)
    line = re.sub("cannot", "can not", line)
    line = re.sub("gonna", "gon na", line)
    line = re.sub("gotta", "got ta", line)
    line = re.sub("wanna", "wan na", line)
    line = re.sub("wan nabe", "wannabe", line)
    line = re.sub("`", " ` ", line)
    line = re.sub(" 'more", " ' more", line)
    line = re.sub("CAN ' T", "CA N'T", line)
    line = re.sub("DIDN ' T", "DID N'T", line)
    line = re.sub("D ' Onofio", "D'Onofio", line)
    line = re.sub("O ' Donnells", "O'Donnells", line)
    line = re.sub("O ' Brien", "O'Brien", line)
    line = re.sub("O ' Brian", "O'Brian", line)
    line = re.sub("d ' oeuvers", "d'oeuvers", line)
    line = re.sub("ma ' am", "ma'am", line)
    line = re.sub("O ' clock", "O'clock", line)
    line = re.sub("o ' clock", "o'clock", line)
    line = re.sub(r"(\d)am", r"\1 am", line)
    line = re.sub(r"(\d)pm", r"\1 pm", line)
    line = re.sub(r"(\d)lbs", r"\1 lbs", line)
    line = re.sub("y ' all", "y' all", line)
    return line


def get_show_name(vid_name):
    """
    get tvshow name from vid_name
    :param vid_name: video clip name
    :return: tvshow name
    """
    show_list = ["friends", "met", "castle", "house", "grey"]
    vid_name_prefix = vid_name.split("_")[0]
    show_name = vid_name_prefix if vid_name_prefix in show_list else "bbt"
    return show_name


def l2_normalize_numpy_array(arr, eps=1e-12, p=2, axis=1):
    """Normalize numpy array (N, D) in D dim,
    this is the same implementation as PyTorch's F.normalize(X, p=2, dim=1)"""
    assert len(arr.shape) == 2
    norm = np.sqrt(np.sum(arr ** p, axis=axis, keepdims=True))
    norm = np.maximum(norm, eps)
    arr = arr / norm
    return arr


def hdf5_to_dict(h5_path):
    print("Loading h5py file into dict ...")
    data = {}
    with h5py.File(h5_path, "r") as h5f:
        for k in tqdm(h5f.keys()):
            data[k] = h5f[k][:]
    return data


class AverageMeter(object):
    """Computes and stores the average and current/max/min value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10

    def update(self, val, n=1):
        self.max = max(val, self.max)
        self.min = min(val, self.min)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
    return n_all, n_trainable


def get_q_type(questions, word2idx):
    qtypes = ["what", "who", "where", "how", "why", "other"]
    indexed_q_types = [word2idx[q_type] for q_type in qtypes]
    q_type_by_example = ["other"] * len(questions)
    for i in range(len(questions)):
        for j in range(len(qtypes)-1):  # ignore the last one
            if questions[i][0] == indexed_q_types[j]:
                q_type_by_example[i] = qtypes[j]
                break
    return q_type_by_example


def compute_acc_by_type(q_types, corrects):
    """
    Args:
        q_types (list of str), q_type for each example
        corrects (list of int), 1/0 predition for each example
    return:
        acc_by_type: list of stringfied acc for each type
    """
    qtypes = ["what", "who", "where", "how", "why", "other"]
    corrects_dict = {}
    for t in qtypes:
        corrects_dict[t] = []
    for i in range(len(q_types)):
        for t in qtypes:
            if q_types[i] == t:
                corrects_dict[t].append(corrects[i])
                break

    acc_by_type = {}
    for t in qtypes:
        acc_by_type[t] = {
            "acc": sum(corrects_dict[t]) * 1.0 / len(corrects_dict[t]) if len(corrects_dict[t]) != 0 else 0.,
            "num_qa": len(corrects_dict[t])
        }
    return acc_by_type


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


def computeIoU(box1, box2):
    """
    :param box1:  [bottom-left-x, bottom-left-y, top-right-x, top-right-y]
    :param box2:  [bottom-left-x, bottom-left-y, top-right-x, top-right-y]
    :return:
    """
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return float(inter)/union

