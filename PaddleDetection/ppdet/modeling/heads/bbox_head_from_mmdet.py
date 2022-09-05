import paddle
import paddle.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32 # 运行方式，为了高效训练,能不能跑起来都不一定，我先注释掉了
# from torch.nn.modules.utils import _pair
from ..utils_pair import _pair
# from mmdet.core import multi_apply#, multiclass_nms ,build_bbox_coder
#from mmdet.models.builder import build_loss，HEADS
# build_xxx 好像是和config，dict访问方式，有关，不知道咋整，先注释掉了
from ppdet.core.workspace import register # 再去看看bbox的注册操作吧
from ..ops import multiclass_nms
from mmdet.models.losses import accuracy
import numpy as np
from ppdet.modeling.losses import SmoothL1Loss

from functools import partial
from six.moves import map, zip
from .file.roiheads.utils import new_zeros, new_ones, new_tensor, view, type, get_shape, new_full
from .file.roiheads.DeltaXYWHBBoxCoder import DeltaXYWHBBoxCoder
from .file.roiheads.loss import CrossEntropyLoss

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

@register
class BBoxHeadDetPro(nn.Layer):
    """BBoxHeadDetPro box head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,
                 bbox_coder=DeltaXYWHBBoxCoder(target_means=[0., 0., 0., 0.],target_stds=[0.1, 0.1, 0.2, 0.2],clip_border=True),
                 # dict(
                 #     type='DeltaXYWHBBoxCoder',
                 #     clip_border=True,# 此clip非彼clip啊
                 #     target_means=[0., 0., 0., 0.],
                 #     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_class_agnostic=False,
                 reg_decoded_bbox=False,
                 loss_cls=CrossEntropyLoss(loss_weight=1.0,use_sigmoid=False),
                 # dict(
                 #     type='CrossEntropyLoss',
                 #     use_sigmoid=False,
                 #     loss_weight=1.0),
                 loss_bbox=SmoothL1Loss(beta=1.0,loss_weight=1.0),
                   # dict(
                   #   type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
    ):
        super(BBoxHeadDetPro, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        # self.with_cls = build_loss(with_cls) 
        # self.with_reg = build_loss(with_reg)
        self.roi_feat_size = (roi_feat_size,roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.fp16_enabled = False

        self.bbox_coder = bbox_coder
        self.loss_cls = loss_cls
        self.loss_bbox =loss_bbox

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2D(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            # need to add background class
            weight_attr = paddle.framework.ParamAttr(
                name="fc_cls.weight",
                initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.01))

            # nn.init.constant_(self.fc_cls.bias, 0)
            bias_attr = paddle.ParamAttr(
                name="bias",
                initializer=paddle.nn.initializer.Constant(value=0))
            self.fc_cls = nn.Linear(in_channels, num_classes + 1,weight_attr= weight_attr,bias_attr=bias_attr)
        if self.with_reg:
            # nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            weight_attr = paddle.framework.ParamAttr(
                name="fc_reg.weight",
                initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.001))
            # nn.init.constant_(self.fc_reg.bias, 0)
            bias_attr = paddle.ParamAttr(
                name="bias",
                initializer=paddle.nn.initializer.Constant(value=0))
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg, weight_attr=weight_attr,bias_attr=bias_attr)
        self.debug_imgs = None

    # def init_weights(self):
    #     # conv layers are already initialized by ConvModule
    #     if self.with_cls:
    #         # nn.init.normal_(self.fc_cls.weight, 0, 0.01)
    #         self.fc_cls.weight = paddle.framework.ParamAttr(
    #             name="fc_cls.weight",
    #             initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.01))
    #
    #         # nn.init.constant_(self.fc_cls.bias, 0)
    #         self.fc_cls.bias = paddle.ParamAttr(
    #             name="bias",
    #             initializer=paddle.nn.initializer.Constant(value=0))
    #     if self.with_reg:
    #         # nn.init.normal_(self.fc_reg.weight, 0, 0.001)
    #         self.fc_cls.weight = paddle.framework.ParamAttr(
    #             name="fc_cls.weight",
    #             initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.001))
    #         # nn.init.constant_(self.fc_reg.bias, 0)
    #         self.fc_cls.bias = paddle.ParamAttr(
    #             name="bias",
    #             initializer=paddle.nn.initializer.Constant(value=0))

    # @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        # x = view(x, (x.size(0), -1))
        x = view(x, (get_shape(x, 0), -1))
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, train_cfg_pos_weight):
        # num_pos = pos_bboxes.size(0)
        num_pos = get_shape(pos_bboxes, 0)
        # num_neg = neg_bboxes.size(0)
        num_neg = get_shape(neg_bboxes, 0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = new_full((num_samples, ),
                            self.num_classes,pos_bboxes,dtype=paddle.int64
                                     )
        label_weights = new_zeros(num_samples, pos_bboxes)
        # label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = new_zeros((num_samples, 4), pos_bboxes)
        # bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = new_zeros((num_samples, 4), pos_bboxes)
        # bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if train_cfg_pos_weight <= 0 else train_cfg_pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg_pos_weight,
                    concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg_pos_weight)

        if concat:
            labels = paddle.concat(x=labels, axis=0)
            label_weights = paddle.concat(x=label_weights, axis=0)
            bbox_targets = paddle.concat(x=bbox_targets, axis=0)
            bbox_weights = paddle.concat(x=bbox_weights, axis=0)
        return labels, label_weights, bbox_targets, bbox_weights

    # @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        # if cls_score is not None:
        #     avg_factor = max(torc-h.sum(label_weights > 0).float().item(), 1.)
        #     if cls_score.numel() > 0:
        #         losses['loss_cls'] = self.loss_cls(
        #             cls_score,
        #             labels,
        #             label_weights,
        #             avg_factor=avg_factor,
        #             reduction_override=reduction_override)
        #         losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = view(bbox_pred,
                        (get_shape(bbox_pred, 0)))[type(pos_inds, paddle.bool)]
                else:
                    pos_bbox_pred = view(
                        bbox_pred,(get_shape(bbox_pred, 0), -1,
                        4))[type(pos_inds, paddle.bool),
                           labels[type(pos_inds, paddle.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[type(pos_inds, paddle.bool)],
                    bbox_weights[type(pos_inds, paddle.bool)],
                    # avg_factor=bbox_targets.size(0),
                    avg_factor=get_shape(bbox_targets, 0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()# 原来只要是paddle有的函数，也可以直接x.sum()
        return losses

    # @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   nms_score_thr=None,
                   nms_thr=None,
                   nms_topk=None,
                   rescale=False,
                   ):
        # if isinstance(cls_score, list):
        #     cls_score = sum(cls_score) / float(len(cls_score))
        # scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        scores = cls_score

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = paddle.clone(rois[:, 1:])
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and get_shape(bboxes, 0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = new_tensor(scale_factor, bboxes)
                # bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                #           scale_factor).view(bboxes.size()[0], -1)
                bboxes = view((view(bboxes, (get_shape(bboxes,0), -1, 4)) /
                               scale_factor), (get_shape(bboxes, 0), -1))

        if nms_score_thr is None or nms_thr is None or nms_topk is None:
            return bboxes, scores
        else:
            # det_bboxes, det_labels = multiclass_nms(bboxes, scores,
            #                                         nms_score_thr, nms_thr,
            #                                         nms_topk)
            output, nms_bbox = multiclass_nms(bboxes, scores,
                                                    nms_score_thr, nms_thr,
                                                    nms_topk)
            det_bboxes = output[:, 1:6]
            det_labels = output[:, 0]
            return det_bboxes, det_labels,nms_bbox


    # @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image. The first column is
                the image id and the next 4 columns are x1, y1, x2, y2.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import kwarray
            >>> import numpy as np
            >>> from mmdet.core.bbox.demodata import random_boxes
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            >>> img_metas = [{'img_shape': (scale, scale)}
            ...              for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torc-h.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torc-h.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torc-h.randint(0, 2, (n_roi,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> # For each image, pretend random positive boxes are gts
            >>> is_label_pos = (labels.numpy() > 0).astype(np.int)
            >>> lbl_per_img = kwarray.group_items(is_label_pos,
            ...                                   img_ids.numpy())
            >>> pos_per_img = [sum(lbl_per_img.get(gid, []))
            ...                for gid in range(n_img)]
            >>> pos_is_gts = [
            >>>     torc-h.randint(0, 2, (npos,)).byte().sort(
            >>>         descending=True)[0]
            >>>     for npos in pos_per_img
            >>> ]
            >>> bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
            >>>                    pos_is_gts, img_metas)
            >>> print(bboxes_list)
        """
        # img_ids = rois[:, 0].long().unique(sorted=True)
        tmp = paddle.floor(rois[:, 0])
        img_ids = paddle.unique(tmp)
        # assert img_ids.numel() <= len(img_metas)
        assert paddle.numel(img_ids) <= len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = paddle.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(axis=1)
            # num_rois = inds.numel()
            num_rois = paddle.numel(inds)

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = new_ones(num_rois, pos_is_gts_)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[type(keep_inds, paddle.bool)]) #bboxes_list是一个list，当然可以append

        return bboxes_list

    # @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert get_shape(rois, 1) == 4 or get_shape(rois, 1) == 5, repr(paddle.shape(rois))# 返回值可能有些问题，就是我感觉返回值除了最后的形状还有前面的八股

        if not self.reg_class_agnostic:
            label = label * 4
            inds = paddle.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = paddle.gather(bbox_pred, inds, 1)  # 這裏可能會出問題，gather上了paddle的幫助文檔了
        assert get_shape(bbox_pred, 1) == 4

        if get_shape(rois, 1) == 4:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=img_meta['img_shape'])
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_meta['img_shape'])
            new_rois = paddle.concat((rois[:, [0]], bboxes), axis=1)

        return new_rois
