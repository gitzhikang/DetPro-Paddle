import os, sys
import math
import random
import numpy as np
import paddle
import paddle.nn.functional as F
from classname import *
import coop_mini
from trainer import test_embedding, test_embedding_neg, train_epoch, get_embedding, checkpoint, accuracy1, accuracy5
from paddle.io import TensorDataset, ComposeDataset,DataLoader
from lr_scheduler import build_lr_scheduler

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


def repeat(input, repeatList):
    if (len(repeatList)== 1):
        repeat_factor = repeatList[0]
        res = np.zeros(shape=[repeat_factor * input.shape[0]], dtype="float32")
        print(res.shape[0])
        for i in range(res.shape[0]):
            res[i] = input[i % input.shape[0]].item()

    elif (len(repeatList) == 2):
        repeat_factor_d0 = repeatList[0]
        repeat_factor_d1 = repeatList[1]
        res = np.zeros(shape=[repeat_factor_d0 * input.shape[0], repeat_factor_d1 * input.shape[1]], dtype="float32")
        print(res.shape)
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res[i][j] = input[i % input.shape[0]][j % input.shape[1]].item()

    res = paddle.to_tensor(res)
    return res



def sample(data, k, cats):
    feat, label, iou = data
    featk, labelk, iouk = [], [], []
    for i in cats:
        id = (label == i)
        if id.sum() == 0:
            continue
        repeat_factor = math.ceil(k / id.sum())
        # print(repeat_factor, id.sum(), feat[id].repeat([repeat_factor, 1]).shape)
        if repeat_factor == 1:
            ids = random.sample(range(id.sum()), k)
            featk.append(feat[id][ids])
            labelk.append(label[id][ids])
            iouk.append(iou[id][ids])
        else:
            # featk.append(feat[id].repeat([repeat_factor - 1, 1]))
            # labelk.append(label[id].repeat([repeat_factor - 1]))
            # iouk.append(iou[id].repeat([repeat_factor - 1]))
            featk.append(repeat(feat[id],[repeat_factor - 1, 1]))
            labelk.append(repeat(label[id],[repeat_factor - 1]))
            iouk.append(repeat(iou[id],[repeat_factor - 1]))

            remain = k - (repeat_factor - 1) * id.sum()
            if remain > 0:
                ids = random.sample(range(id.sum()), remain)
                featk.append(feat[id][ids])
                labelk.append(label[id][ids])
                iouk.append(iou[id][ids])

    featk = paddle.concat(featk, axis=0)
    labelk = paddle.concat(labelk, axis=0)
    iouk = paddle.concat(iouk, axis=0)
    return featk, labelk, iouk


if __name__ == "__main__":
    # torch.random.manual_seed(825)
    # random.seed(825)

    print(sys.argv)
    _, mode, train_dir, val_dir, res_dir, prefix, mode_train, bg_thr, iou_lb, iou_ub = sys.argv[:10]
    if mode == "train":
        if len(sys.argv) == 10:
            ctx_num = 8
            cls_token_position = 'end'
            neg_split = 10
        elif len(sys.argv) == 11:
            ctx_num = int(sys.argv[10])
            cls_token_position = 'end'
            neg_split = 10
        elif len(sys.argv) == 12:
            ctx_num = int(sys.argv[10])
            cls_token_position = sys.argv[11]
            neg_split = 10
        else:
            ctx_num = int(sys.argv[10])
            cls_token_position = sys.argv[11]
            neg_split = sys.argv[12]
    # else:
    # _, mode, train_dir, val_dir, res_dir, prefix, mode_train, bg_thr, iou_lb, iou_ub,ctx_num = sys.argv[:11]
    if mode == "test":
        ctx_num = 8
        cls_token_position = 'end'
        neg_split = 10
        names = sys.argv[10:]
    bg_thr, iou_lb, iou_ub = float(bg_thr), float(iou_lb), float(iou_ub)

    # Clip
    clip_model = coop_mini.load_clip_to_cpu().float()
    # Clip不使用梯度
    for params in clip_model.parameters():
        params.stop_gradient=True
    model = coop_mini.CustomCLIP(CLASS_NAMES, clip_model, True, bg_class=(mode_train == 'learn'), ctx=ctx_num,
                                 cls_token_position=cls_token_position).to('cuda')
    print('MODEL BUILD COMPLETE')

    if mode == 'train':
        train_base, train_set_novel, train_neg = load_data_new(os.path.join(train_dir, 'train_data.pth'), iou_lb, True)
        train_base = data_iou_filter(train_base, iou_lb, iou_ub)
        train_neg = data_iou_filter(train_neg, 0.1, bg_thr)
        # train_neg = data_iou_filter(train_neg, 0.1, 1.1)

        # k = len(train_base[0]) // 866
        # train_base = sample(train_base, len(train_base[0]//int(neg_split)), range(866))
        train_neg = sample(train_neg, len(train_neg[0]) // int(neg_split), [866])

        train_base, train_set_novel, train_neg = TensorDataset(*train_base), TensorDataset(
            *train_set_novel), TensorDataset(*train_neg)
        train_data = train_base if (mode_train == 'fg_only') else ComposeDataset([train_base, train_neg])
        print("train info:", len(train_base), len(train_set_novel), len(train_neg))

        freq = get_freq(train_base)
        freq = [x / len(train_data) * 866 for x in freq]
        freq = freq[:866]
        freq.append(2 * len(train_neg) / len(train_data))  # background
        # freq = get_freq(train_base)
        # print("min freq =", min(freq[:866]))
        # freq = [x / 5e5 * 866 for x in freq]
        # freq = freq[:866]
        # freq.append(2 * len(train_neg) / 5e5) # background
        # print(k, k/4e5*866)
        # actual_cls = len(train_base) // k
        # print(actual_cls)
        # freq = [k/4e5*actual_cls] * 866 + [2*len(train_neg)/4e5]
        # freq = [1] * 867
        # print(freq)

        train_dl = DataLoader(train_data, batch_size=512, shuffle=True)

    if mode == 'test':
        val_base, val_novel, _ = load_data_new(os.path.join(val_dir, 'val_data.pdparams'))
        _, _, val_neg = load_data_new(os.path.join(val_dir, 'val_data.pdparams'), iou_lb)
        val_base = data_iou_filter(val_base, iou_ub, 1.1)
        val_novel = data_iou_filter(val_novel, iou_ub, 1.1)
        val_neg = data_iou_filter(val_neg, bg_thr, iou_lb)
    else:
        # val_base = [feature,label,iou]
        val_base, val_novel, val_neg = load_data_new(os.path.join(val_dir, 'val_data..pdparams'))

    val_base, val_novel, val_neg = TensorDataset(*val_base), TensorDataset(*val_novel), TensorDataset(*val_neg)
    print("val info:", len(val_base), len(val_novel), len(val_neg))
    val_dl1 = DataLoader(val_base, batch_size=1024, shuffle=True)
    val_dl2 = DataLoader(val_novel, batch_size=1024, shuffle=True)
    # val_dlp = DataLoader(train_set_novel, batch_size = 1024, shuffle=True)
    neg_val_dl = DataLoader(val_neg, batch_size=1024, shuffle=True)

    if mode == 'train':

        emb = get_embedding(model, CLASS_NAMES_FULL)
        test_embedding(emb, val_dl1)
        test_embedding(emb, val_dl2)
        # test_neg(emb)

        os.makedirs(res_dir, exist_ok=True)
        # optimizer = SGD(model.prompt_learner.parameters(), lr=2e-3)
        scheduler = build_lr_scheduler(2e-3, 6, 0, 0)
        optimizer = paddle.optimizer.SGD(parameters=model.parameters(),learning_rate=scheduler)


        for i in range(6):
            print(f"epoch{i + 1}")
            train_epoch(model, optimizer, train_dl, freq, mode_train)
            emb = get_embedding(model, CLASS_NAMES_FULL)
            print('val on base')
            test_embedding(emb, val_dl1)
            print('val on novel')
            test_embedding(emb, val_dl2)
            test_neg(emb)
            if emb.shape[0] > 1203:
                test_embedding(emb[:1203], val_dl1)
                test_embedding(emb[:1203], val_dl2)
                test_neg(emb[:1203])
            scheduler.step()
            checkpoint(model, os.path.join(res_dir, prefix + f"epoch{i + 1}"), CLASS_NAMES_FULL)
            checkpoint(model, os.path.join(res_dir, prefix + f"epoch{i + 1}_empty"), [""])

        checkpoint(model, os.path.join(res_dir, prefix + f"epoch{i + 1}"), CLASS_NAMES_FULL)
        checkpoint(model, os.path.join(res_dir, prefix + f"coco"), COCO_CLASSES)
        checkpoint(model, os.path.join(res_dir, prefix + f"voc"), VOC_CLASSES)
        checkpoint(model, os.path.join(res_dir, prefix + f"objects365"), Objects365_CLASSES)

    elif mode == 'test':
        # Ensemble embedding
        if names == None:
            # names = ['lvis_text_embedding.pt']
            names = ['checkpoints/exp1/test_epoch6.pdparams']
            # names = ['checkpoints/optim/seed456_hinge_adam_warmup_epoch7.pth']
            # names = [f'checkpoints/gen6_ens/pos{i}epoch6.pth' for i in [5,6,7,8,9]]
            # names = [f'checkpoints/gen6_ens/pos{i}voc.pth' for i in [5,6,7,8,9]]
            # names = [f'checkpoints/gen_ens/pos{i}epoch2.pth' for i in [5,6,7,8,9]]
            # names = [f'checkpoints/pos_ens/0{i}_epoch6.pth' for i in [5,6,7,8,9]]
        # if use_weight:
        ensemble_embedding = load_ens_embedding(names, norm=True)  # / 5#[:1203]
        # ensemble_embedding = load_ens_embedding(names, norm = True)# / 5#[:1203]
        paddle.save(ensemble_embedding, os.path.join(res_dir, prefix + "_ens.pdparams"))

        print("loaded ensemble embedding :", ensemble_embedding.shape)
        # test_embedding(ensemble_embedding[lvis_base_label_ids], train_dl)
        test_embedding(ensemble_embedding, val_dl1)
        test_embedding(ensemble_embedding, val_dl2)
        # test_embedding(ensemble_embedding, neg_val_dl)
        # test_embedding(ensemble_embedding, val_dlp)
        test_neg(ensemble_embedding)

        if ensemble_embedding.shape[0] > 1203:
            print('1203 only')
            ensemble_embedding = ensemble_embedding[:1203]
            # test_embedding(ensemble_embedding[lvis_base_label_ids], train_dl)
            test_embedding(ensemble_embedding, val_dl1)
            test_embedding(ensemble_embedding, val_dl2)
            # test_embedding(ensemble_embedding, val_dlp)
            test_neg(ensemble_embedding)
            # test_embedding(ensemble_embedding, val_dlp_gt)

    elif mode == 'multi':
        names = [f'checkpoints/gen6_ens/pos{i}epoch6.pdparams' for i in [5, 6, 7, 8, 9]]
        iou_embedding = [load_ens_embedding([name], norm=True)[:1203] for name in names]


        # iou_embedding = torch.cat(iou_embedding)

        def test_embedding_iou(embedding, ds):
            acc1, acc5 = 0, 0
            iter = 0
            iou_thr = [.6, .7, .8, .9, 1.1]
            for feat, label, iou in ds:
                iter += 1
                if iter % 10 == 0:
                    print(iter, '/', len(ds), end='\r')

                # res = feat.to('cuda') @ embedding.t() / 0.01
                res = [feat.cuda() @ emb.t() / 0.01 for emb in embedding]
                res = paddle.concat(res, axis=1)
                # print(res.shape)
                # res[:,lvis_base_label_ids] = -1e10
                for id, cur in enumerate(iou):
                    for tm, key in enumerate(iou_thr):
                        if cur < key:
                            break
                    label[id] = label[id] + 1203 * tm
                # print(label)
                acc1 += accuracy1(res.cpu(), label)
                acc5 += accuracy5(res.cpu(), label)

            acc1 = acc1.item() / len(ds.dataset)
            acc5 = acc5.item() / len(ds.dataset)
            print(f"test acc: top1={acc1}   top5={acc5}   total={len(ds.dataset)}")


        test_embedding_iou(iou_embedding, val_dl1)
        test_embedding_iou(iou_embedding, val_dl2)
    else:
        print('unknown mode')