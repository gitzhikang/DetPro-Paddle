# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['VildDetpro']


@register
class VildDetpro(BaseArch):
    """
    Mask R-CNN network, see https://arxiv.org/abs/1703.06870

    Args:
        backbone (object): backbone instance
        rpn_head (object): `RPNHead` instance
        bbox_head (object): `BBoxHead` instance
        mask_head (object): `MaskHead` instance
        bbox_post_process (object): `BBoxPostProcess` instance
        mask_post_process (object): `MaskPostProcess` instance
        neck (object): 'FPN' instance
    """

    __category__ = 'architecture'
    __inject__ = [
        'bbox_post_process',
        'mask_post_process',
    ]

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_post_process,
                 mask_post_process,
                 roi_head,
                 neck=None):
        super(MaskRCNN, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.bbox_post_process = bbox_post_process
        self.mask_post_process = mask_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = cfg['neck'] and create(cfg['neck'], **kwargs)

        out_shape = neck and neck.out_shape or backbone.out_shape
        kwargs = {'input_shape': out_shape}
        rpn_head = create(cfg['rpn_head'], **kwargs)
        # bbox_head = create(cfg['bbox_head'], **kwargs)
        roi_head = create(cfg['roi_head'],**kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "rpn_head": rpn_head,
            "roi_head":roi_head
        }

    def _forward(self):
        img_metas = {
            'filename': self.inputs['im_file'],
            'ori_filename': self.inputs['ori_filename'],
            'ori_shape': (self.inputs['h'], self.inputs['w'], 3),
            'img_shape': (self.inputs['image'].shape[2],self.inputs['image'].shape[3],3),
            'pad_shape': self.inputs['pad_shape'],
            'scale_factor': self.inputs['scale_factor'],
            'flip': self.inputs['flipped'],
            'flip_direction': self.inputs['flip_direction'],
            'img_norm_cfg': self.inputs['img_norm_cfg']
        }

        body_feats = self.backbone(self.inputs)
        if self.neck is not None:
            body_feats = self.neck(body_feats)


        if self.training:
            rois, rois_num, rpn_loss = self.rpn_head(body_feats, self.inputs)

            # bbox_loss, bbox_feat = self.bbox_head(body_feats, rois, rois_num,
            #                                       self.inputs)
            # rois, rois_num = self.bbox_head.get_assigned_rois()
            # bbox_targets = self.bbox_head.get_assigned_targets()
            # # Mask Head needs bbox_feat in Mask RCNN
            # mask_loss = self.mask_head(body_feats, rois, rois_num, self.inputs,
            #                            bbox_targets, bbox_feat)

            losses = self.roi_head.forward_train(body_feats,self.inputs['image'],self.inputs['img_no_normalize'],img_metas,rois,self.inputs['pre_computed_proposal'],
                                        self.inputs['gt_bbox'],self.inputs['gt_class'],self.inputs['bboxes_ignore'],self.inputs['gt_poly']
                                        )
            return rpn_loss, losses
        else:
            rois, rois_num, _ = self.rpn_head(body_feats, self.inputs)
            # preds, feat_func = self.bbox_head(body_feats, rois, rois_num, None)
            #
            # im_shape = self.inputs['im_shape']
            # scale_factor = self.inputs['scale_factor']
            #
            # bbox, bbox_num = self.bbox_post_process(preds, (rois, rois_num),
            #                                         im_shape, scale_factor)
            # mask_out = self.mask_head(
            #     body_feats, bbox, bbox_num, self.inputs, feat_func=feat_func)
            #
            # # rescale the prediction back to origin image
            # bbox, bbox_pred, bbox_num = self.bbox_post_process.get_pred(
            #     bbox, bbox_num, im_shape, scale_factor)
            # origin_shape = self.bbox_post_process.get_origin_shape()
            # mask_pred = self.mask_post_process(mask_out, bbox_pred, bbox_num,
            #                                    origin_shape)
            det_bboxes,det_labels,segm_results,det_nums=self.roi_head.simple_test(body_feats,self.inputs['image'],
                                      self.inputs['img_no_normalize'],
                                      rois,img_metas,
                                      self.inputs['pre_computed_proposal'],
                                        True
                                        )
            bboxes = paddle.concat([det_labels,det_bboxes],axis=2)

            return bboxes, det_nums, segm_results

    def __forwar(self):
        body_feats = self.backbone(self.inputs)
        if self.neck is not None:
            body_feats = self.neck(body_feats)
        if self.training:
            rois, rois_num, rpn_loss = self.rpn_head(body_feats, self.inputs)


    def get_loss(self, ):
        rpn_loss, losses = self._forward()
        loss = {}
        loss.update(rpn_loss)
        loss.update(losses)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        bbox_pred, bbox_num, mask_pred = self._forward()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num, 'mask': mask_pred}
        return output
