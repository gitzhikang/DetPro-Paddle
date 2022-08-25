import os, sys
import math
import random

import paddle
import paddle.nn.functional as F
from classname import *
import coop_mini

CLASS_NAMES_FULL = [x.replace('(','').replace(')','').replace('_', ' ') for x in CLASS_NAMES_FULL]

CLASS_NAMES = []
for id in lvis_base_label_ids:
    CLASS_NAMES.append(CLASS_NAMES_FULL[id])


def load_data_new(data_path: str, neg_thr=None, remap=False):
    features, labels, ious = paddle.load(data_path)  # 。。。/train.pth 得是

    if neg_thr == None:
        neg = (labels < 0)
    else:
        neg = (ious < neg_thr)
        labels[neg] = -1

    base = paddle.zeros(shape = labels.shape).astype("bool")
    novel = paddle.zeros(shape = labels.shape).astype("bool")
    # 将base中在lvis_base_label_ids有的类全部变为True
    for i in lvis_base_label_ids: base = paddle.logical_or(labels , i)
    for i in lvis_novel_label_ids: novel = paddle.logical_or(labels , i)

    if remap:
        mapping = paddle.to_tensor(lvis_base_label_ids).astype("int64")
        base_labels = (labels[base].view(-1, 1) == mapping).astype("int32").argmax(axis=1)
        base_data = features[base], base_labels, ious[base]
        neg_data = features[neg], paddle.ones(shape=labels[neg].shape) * 866, ious[neg]
    else:
        base_data = features[base], labels[base], ious[base]
        neg_data = features[neg], labels[neg], ious[neg]
    novel_data = features[novel], labels[novel], ious[novel]

    return base_data, novel_data, neg_data

def data_iou_filter(data, lb, ub):
    features, labels, ious = data
    valid = paddle.logical_and(lb <= ious, ious < ub)
    return features[valid], labels[valid], ious[valid]

def get_freq(data):
    freq = [0] * 1204
    for feat, label, iou in data:
        freq[label] += 1
    return freq

def load_ens_embedding(name_list, norm = False,weight=None):
    emb = [paddle.load(name).astype("float32") for name in name_list]
    if weight is not None:
        emb = [x*w / paddle.linalg.norm(x, p=2, axis=-1,keepdim=True) for x,w in zip(emb,weight)]
    else:
        emb = [x / paddle.linalg.norm(x, p=2, axis=-1,keepdim=True) for x in emb]
    emb = sum(emb)
    emb = emb.squeeze(axis = 0)
    if norm:
        emb = emb / paddle.linalg.norm(emb, p=2, axis=-1,keepdim=True)
    return emb

def test_neg(embedding):
    # return
    for thr in [0.5, 0.9]:
        test_embedding_neg(embedding, neg_val_dl, thr)