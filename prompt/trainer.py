from classname import *
import random
import paddle
from config import temperature
import paddle.nn.functional as F
import time

def get_embedding(model, class_names):
    with paddle.no_grad():
        prompts, tokenized_prompts = model.prompt_learner.forward_for_classes(class_names)
        text_features = model.text_encoder(prompts, tokenized_prompts)

        # text_features = torch.cat([text_features, model.bg_embedding], dim = 0)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features / paddle.linalg.norm(text_features, p=2, axis=-1,keepdim=True)
    return text_features

def checkpoint(model, name, class_names):
    with paddle.no_grad():
        prompts, tokenized_prompts = model.prompt_learner.forward_for_classes(class_names)
        text_features = model.text_encoder(prompts, tokenized_prompts)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # text_features = torch.cat([text_features, model.bg_embedding], dim=0)

        text_features = text_features[None, ...]
        paddle.save(text_features, name+".pdparams")
        prompt = model.prompt_learner.ctx.data
        paddle.save(prompt, name+"_prompt.pdparams")

def accuracy1(logits, labels):
    return paddle.equal(paddle.argmax(logits, axis = -1),labels).sum()
def accuracy5(logits, labels):
    return paddle.equal(logits.astype('float32').topk(k=5)[1],labels[:,None]).sum()

def test_embedding(embedding, ds):
    iter = 0
    acc1, acc5 = 0, 0
    avg_score, avg_var, entropy = 0, 0, 0

    for feat, label, iou in ds:
        iter += 1
        if iter % 100 == 0:
            print(iter, '/', len(ds), end='\r')

        res = feat.cuda() @ embedding.t() / temperature
        res = F.softmax(res, axis=-1)
        # res[:,lvis_base_label_ids] = -1e10
        acc1 += accuracy1(res.cpu(), label)
        acc5 += accuracy5(res.cpu(), label)

        # avg_score += res.max(dim = -1)[0][:1203].sum()
        # avg_var += res.var(dim=-1)[:1203].sum()
        # entropy += -res.log().sum(dim=-1)[:1203].sum()
        res = res[:,:1203]
        avg_score += res.max(axis = -1)[0].sum()
        avg_var += res.var(axis=-1).sum()
        entropy += (-res.log() * res).sum()

    acc1 = acc1.item() / len(ds.dataset)
    acc5 = acc5.item() / len(ds.dataset)
    avg_score = avg_score.item() / len(ds.dataset)
    avg_var = avg_var.item() / len(ds.dataset)
    entropy = entropy.item() / len(ds.dataset)
    print(f"test acc: top1={acc1}   top5={acc5}   total={len(ds.dataset)}")
    print(f"avg_score: {avg_score}      avg_var: {avg_var}     entropy: {entropy}")
def test_embedding_neg(embedding, ds, thr):
    iter = 0
    pos = 0
    avg_score, avg_var, entropy = 0, 0, 0

    for feat, label, iou in ds:
        iter += 1
        if iter % 100 == 0:
            print(iter, '/', len(ds), end='\r')

        res = feat.cuda() @ embedding.t() / temperature
        res = F.softmax(res, axis=-1)[:,:1203]
        # print(res)
        pos += (res.max(axis=-1)[0] >= thr).sum()

        avg_score += res.max(axis = -1)[0].sum()
        avg_var += res.var(axis=-1).sum()
        entropy += (-res.log() * res).sum()

    avg_score = avg_score.item() / len(ds.dataset)
    avg_var = avg_var.item() / len(ds.dataset)
    entropy = entropy.item() / len(ds.dataset)
    print(f"test neg(thr={thr}): pos={pos}  total={len(ds.dataset)}")
    print(f"avg_score: {avg_score}      avg_var: {avg_var}     entropy: {entropy}")

def scatter(input,index,src):
    for i in range(index.shape[0]):
        for j in range(index.shape[1]):
            input[i][index[i][j]] = src[i][j]

def softLabel(label, iou):
    softlb = paddle.zeros(shape=[label.shape[0], 867])
    # softlb = iou.new_zeros((label.shape[0], 867))
    #相当于给原来向量多增加一个维度
    softlb = scatter(softlb,label[:,None].long(), iou[:,None])
    softlb[label!=866,-1] = 1-iou[label!=866]
    softlb[label==866] = 1/867
    return softlb

def softCrossEntropy(logit, target):
    log_likelihood = -F.log_softmax(logit, axis = 1)
    return paddle.sum(log_likelihood * target)


def train_epoch(model, optimizer, ds, freq=None, mode='hinge'):
    print("train mode :", mode)
    with paddle.no_grad():
        emb = model.get_embedding()
        print("embbeding shape :", emb.shape)

    acc1, acc5 = 0, 0
    idb = 0
    time_s = time.time()
    # 半精度加快训练
    with paddle.amp.auto_cast():
        for feat, label, iou in ds:
            idb += 1
            if idb % 100 == 0:
                print(idb, '/', len(ds), end='  ')

            emb = model.get_embedding()[:867]
            # emb.norm(dim=-1, keepdim=True)
            emb = emb / paddle.linalg.norm(emb, p=2, axis=-1,keepdim=True)
            # emb = torch.cat([embr[:866] / embr[:866].norm(dim = -1, keepdim = True), embr[-1:]])
            # sim = emb @ emb.T - torch.eye(emb.shape[0]).cuda()
            # print(sim)
            # print(sim.shape)
            # print((sim + torch.eye(emb.shape[0]).cuda()).min(), sim.max(), sim.mean())
            # # input()
            # closs = (sim-0.8).maximum(torch.tensor(0).cuda()).sum()

            feat = feat.cuda()
            # feat.norm(dim=-1, keepdim=True)
            feat = feat / paddle.linalg.norm(feat, p=2, axis=-1,keepdim=True)
            res = feat @ emb.t() / temperature

            # res = model(feat.to('cuda')) / temperature
            # print(res.shape)
            acc1 += accuracy1(res.cpu(), label)
            acc5 += accuracy5(res.cpu(), label)

            weight = 1 / paddle.to_tensor(freq).cuda()
            if mode == 'hinge':  # hinge loss
                res = res[:, :866]
                loss = F.cross_entropy(res[label.cuda() != 866], label.cuda()[label.cuda() != 866], weight[:866],
                                       reduction="sum")

                logit = F.softmax(res, axis=-1)
                bg_loss = (logit - (1 / 866)).maximum(paddle.to_tensor(0).cuda()).sum(axis=-1)
                bg_loss = bg_loss[label.cuda() == 866].sum() * weight[866]
                loss += bg_loss
                loss /= len(label)
                # loss /= weight[label].sum()

            if mode == 'mean':  # mean loss
                res = res[:, :866]
                res = paddle.concat([res, res.mean(dim=-1, keepdim=True)], axis=-1)
                loss = F.cross_entropy(res, label.cuda(), weight)

            if mode == 'meanbg':  # mean loss only on bg
                res = res[:, :866]
                fg = label.cuda() < 866
                loss = F.cross_entropy(res[fg], label.cuda()[fg], weight[:866], reduction="sum")
                res = paddle.concat([res, res.mean(dim=-1, keepdim=True)], axis=-1)
                bg = label.cuda() == 866
                loss_bg = F.cross_entropy(res[bg], label.cuda()[bg], weight, reduction="sum")
                loss = (loss + loss_bg) / len(label)

            if mode == 'max':  # max loss
                res = res[:, :866]
                alpha = 1.
                res = paddle.concat([res, 1 - alpha * res.max(dim=-1, keepdim=True)[0]], axis=-1)
                res = F.softmax(res, axis=-1)
                loss = F.nll_loss(res.log(), label.cuda(), weight)

            if mode == 'learn':  # learn a bg embedding
                loss = F.cross_entropy(res, label.cuda(), weight)
                # loss *= weight[0]
                # loss2 = F.cross_entropy(res, label.cuda(), reduction="mean")
                # print(weight)
                # print(loss2/loss)
            if mode == 'fg_only':  # learn a bg embedding
                loss = F.cross_entropy(res, label.cuda(), weight[:866])
                # loss /= weight[label].sum()

            if mode == 'soft':  # soft label with 1/C
                # soft_label = softLabel(label.cuda(), iou.cuda())
                res = res[:, :866]
                fg = label.cuda() < 866
                loss_fg = F.cross_entropy(res[fg], label.cuda()[fg], weight[:866], reduction="sum")

                bg = label.cuda() == 866
                # soft_label = res.new_ones(res.shape).cuda() / 866
                soft_label = paddle.ones(shape=res.shape).cuda() / 866
                loss_bg = softCrossEntropy(res[bg], soft_label[bg]) * weight[866]
                # loss_bg = F.cross_entropy(res[bg], label.cuda()[bg], weight, reduction="sum")
                loss = (loss_fg + loss_bg) / weight.cuda()[label.cuda()].sum()
                # loss *= weight[0]
                # coef =
                # loss = (coef * loss).mean()

            if idb % 100 == 0:
                print("loss =", loss, f"  time={time.time() - time_s}", end='\r')
                time_s = time.time()
            model.clear_grad()

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

        acc1 = acc1.item() / len(ds.dataset)
        acc5 = acc5.item() / len(ds.dataset)
        print(f"train acc: top1={acc1}   top5={acc5}   total={len(ds.dataset)}")