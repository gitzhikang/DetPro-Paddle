import numpy as np
from class_name import *
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, XavierUniform, KaimingNormal
from paddle.regularizer import L2Decay
from passl.modeling.architectures import CLIPWrapper
from passl.modeling.backbones import clip
from ppdet.core.workspace import register, create
from .roi_extractor import RoIAlign
from ..shape_spec import ShapeSpec
from ..bbox_utils import bbox2delta
from ppdet.modeling.layers import ConvNormLayer
import time
from .zip import ZipBackend
from .file.roiheads.file_client import FileClient
from tqdm import tqdm
import os.path as osp
from prompt.coop_mini import tokenize
from lvis import LVIS
import pickle
import io
from PIL import Image
import os
from paddle.vision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from passl.datasets.preprocess.transforms import ToRGB
from .file.roiheads.transforms import bbox2result,bbox2roi,bbox_mapping
from .file.roiheads.merge_augs import merge_aug_bboxes
from .file.roiheads.utils import new_ones
from ..ops import multiclass_nms

def load_clip():
    # url = clip._MODELS["ViT-B/32"]
    # model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))
    #
    # try:
    #     # loading JIT archive
    #     print("jit version")
    #     model = torch.jit.load(model_path, map_location='cpu').eval()
    #     state_dict = None
    #
    # except RuntimeError:
    #     state_dict = torch.load(model_path, map_location='cpu')
    #
    # model = clip.build_model(state_dict or model.state_dict())
    #
    # return model
    arch = {'name': 'CLIP', 'embed_dim': 512,
            'image_resolution': 224, 'vision_layers': 12,
            'vision_width': 768, 'vision_patch_size': 32,
            'context_length': 77, 'vocab_size': 49408,
            'transformer_width': 512, 'transformer_heads': 8,
            'transformer_layers': 12, 'qkv_bias': True, 'proj': True, 'pre_norm': True}
    head = {'name': 'CLIPHead'}
    model = CLIPWrapper(architecture=arch, head=head)
    state_dict = paddle.load("ViT-B-32.pdparams")['state_dict']
    model.set_state_dict(state_dict)
    preprocess = Compose([Resize(224,interpolation='bicubic'),
                        CenterCrop(224),
                        ToRGB(),
                        ToTensor(),
                        Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711],
                                    )
                     ])
    model = model.cuda()
    return model , preprocess

@register
class StandardRoiHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['bbox_assigner', 'bbox_loss']
    """
    RCNN bbox head

    Args:
        head (nn.Layer): Extract feature in bbox head
        in_channel (int): Input channel after RoI extractor
        roi_extractor (object): The module of RoI Extractor
        bbox_assigner (object): The module of Box Assigner, label and sample the 
            box.
        with_pool (bool): Whether to use pooling for the RoI feature.
        num_classes (int): The number of classes
        bbox_weight (List[float]): The weight to get the decode box 
    """

    def __init__(self,
                 bbox_head=None,
                 bbox_roi_extractor=RoIAlign().__dict__,
                 mask_roi_extractor=None,
                 mask_head=None,
                 use_clip_inference=False,
                 load_feature=True,
                 kd_weight=256,
                 fixed_lambda=None,
                 prompt_path=None,
                 coco_setting=False,
                 fix_bg=False,
                 feature_path='data/lvis_clip_image_embedding.zip',
                 test_nms_score_thr=0.0001,
                 test_nms_max_per_img=300,
                 test_nms_iou_threshold=0.5,




                 bbox_assigner='BboxAssigner',
                 with_pool=False,
                 num_classes=80,
                 bbox_weight=[10., 10., 5., 5.],
                 bbox_loss=None,
                 in_channel=None):
        super(StandardRoiHead, self).__init__()
        device = paddle.CUDAPlace(0) if paddle.device.cuda.device_count()>0 else paddle.CPUPlace()
        self.device = device
        if bbox_head.num_classes == 80:
            self.CLASSES = COCO_CLASSES
            dataset = 'coco'
        elif bbox_head.num_classes == 20:
            self.CLASSES = VOC_CLASSES
            dataset = 'voc'
        elif bbox_head.num_classes == 1203:
            self.CLASSES = LVIS_CLASSES
            dataset = 'lvis'
        elif bbox_head.num_classes == 365:
            self.CLASSES = Object365_CLASSES
            dataset = 'objects365'
        self.num_classes = len(self.CLASSES)
        print('num_classes:', self.num_classes)
        if self.num_classes == 1203:
            self.base_label_ids = paddle.to_tensor(lvis_base_label_ids, place=device)
            self.novel_label_ids = paddle.to_tensor(lvis_novel_label_ids, place=device)
            self.novel_index = F.pad(paddle.bincount(self.novel_label_ids),
                                     (0, self.num_classes - self.novel_label_ids.max())).bool()
        elif self.num_classes == 20:
            self.novel_label_ids = paddle.to_tensor(voc_novel_label_ids, device=device)
            self.novel_index = F.pad(paddle.bincount(self.novel_label_ids),
                                     (0, self.num_classes - self.novel_label_ids.max())).bool()
        elif self.num_classes == 80:
            # self.base_label_ids = torch.tensor(coco_base_label_ids, device=device)
            self.novel_label_ids = paddle.to_tensor(coco_unseen_ids_train, device=device)
            self.unseen_label_ids_test = paddle.to_tensor(coco_unseen_ids_test, device=device)

            self.novel_index = F.pad(paddle.bincount(self.novel_label_ids),
                                     (0, self.num_classes - self.novel_label_ids.max())).bool()

        # self.rank = dist.get_rank()
        self.test_nms_score_thr=test_nms_score_thr
        self.test_nms_max_per_img=test_nms_max_per_img
        self.test_nms_iou_threshold=test_nms_iou_threshold

        self.load_feature = load_feature
        self.use_clip_inference = use_clip_inference
        self.kd_weight = kd_weight
        self.fixed_lambda = fixed_lambda
        self.coco_setting = coco_setting
        self.fix_bg = fix_bg
        self.bbox_head=bbox_head
        self.bbox_roi_extractor = bbox_roi_extractor
        self.mask_roi_extractor = mask_roi_extractor,
        self.mask_head = mask_head,
        print('load_feature', load_feature)
        print('use_clip_inference', use_clip_inference)
        print('fixed_lambda', fixed_lambda)
        print('prompt path', prompt_path)
        self.coco_setting = coco_setting
        # self.reporter = MemReporter(self.clip_model)
        self.clip_model,self.preprocess= load_clip()
        self.clip_model.eval()
        for child in self.clip_model.children():
            for param in child.parameters():
                param.stop_gradient=True
        # 看是否图片特征有存储
        if not self.load_feature:
            self.clip_model,self.preprocess= load_clip()
            self.clip_model.eval()
            for child in self.clip_model.children():
                for param in child.parameters():
                    param.stop_gradient=True
        else:
            time_start = time.time()
            self.zipfile = ZipBackend(feature_path)

            # if self.num_classes == 1203:
            #     self.zipfile = ZipBackend('lvis_clip_image_embedding.zip')
            # elif self.num_classes == 80:
            #     self.zipfile = ZipBackend('coco_clip_image_embedding.zip')
            print('load zip:', time.time() - time_start)
        self.text_features_for_classes = []
        self.iters = 0
        self.ensemble = bbox_head.ensemble
        print('ensemble:{}'.format(self.ensemble))
        if prompt_path is not None:
            save_path = prompt_path
        else:
            save_path = 'lvis_text_embedding.pdparams'
        print('load:', save_path)
        time_start = time.time()
        if osp.exists(save_path):
            # if False:
            if not self.fix_bg:
                self.text_features_for_classes = paddle.load(save_path).cuda().squeeze()[:self.num_classes]
            else:
                self.text_features_for_classes = paddle.load(save_path).cuda().squeeze()
                print(self.text_features_for_classes.shape)
        else:
            self.clip_model,self.preprocess=load_clip()
            self.clip_model.eval()
            for child in self.clip_model.children():
                for param in child.parameters():
                    param.stop_gradient=True
            for template in tqdm(template_list):
                print(template)
                text_features_for_classes = paddle.concat(
                    [self.clip_model.encode_text(tokenize(template.format(c)).cuda()).detach() for c in
                     self.CLASSES])
                self.text_features_for_classes.append(F.normalize(text_features_for_classes, axis=-1))

            self.text_features_for_classes = paddle.stack(self.text_features_for_classes).mean(axis=0)
            paddle.save(self.text_features_for_classes.detach().cpu(), save_path)
        self.text_features_for_classes = self.text_features_for_classes.astype('float32')
        self.text_features_for_classes = F.normalize(self.text_features_for_classes, axis=-1)
        print('text embedding finished, {} passed'.format(time.time() - time_start))
        print(self.text_features_for_classes.shape)
        # reporter.report()
        self.proposals = pickle.load('data/lvis_v1/proposals/rpn_r101_fpn_lvis_train.pkl','rb')
        coco = LVIS('data/lvis_v1/annotations/lvis_v1_train.json')
        img_ids = coco.get_img_ids()
        self.file_idxs = dict()
        for i, id in enumerate(img_ids):
            info = coco.load_imgs([id])[0]
            filename = info['coco_url'].replace(
                'http://images.cocodataset.org/', '')
            self.file_idxs[filename] = i
        if not self.fix_bg:
            weight_attr = paddle.ParamAttr(
                name="weight",
                initializer=paddle.nn.initializer.XavierUniform())
            bias_attr = paddle.ParamAttr(
                name="bias",
                initializer=paddle.nn.initializer.Constant(value=0))
            self.bg_embedding = paddle.nn.Linear(1, 512, weight_attr=weight_attr, bias_attr=bias_attr)

        self.temperature = 0.01
        self.accuracy_align = []
        self.accuracy = []
        # self.trans_to_pil = ToPILImage()
        self.color_type = 'color'
        self.file_client = FileClient(backend='disk')
        if self.ensemble:
            # self.projection_for_image = nn.Linear(1024, 512)
            # nn.init.xavier_uniform_(self.projection_for_image.weight)
            # nn.init.constant_(self.projection_for_image.bias, 0)
            weight_attr = paddle.ParamAttr(
                name="weight",
                initializer=paddle.nn.initializer.XavierUniform())
            bias_attr = paddle.ParamAttr(
                name="bias",
                initializer=paddle.nn.initializer.Constant(value=0))
            self.projection_for_image = paddle.nn.Linear(1024, 512, weight_attr=weight_attr, bias_attr=bias_attr)

        # self.projection = paddle.nn.Linear(1024, 512)
        # nn.init.xavier_uniform_(self.projection.weight)
        # nn.init.constant_(self.projection.bias, 0)
        weight_attr = paddle.ParamAttr(
            name="weight",
            initializer=paddle.nn.initializer.XavierUniform())
        bias_attr = paddle.ParamAttr(
            name="bias",
            initializer=paddle.nn.initializer.Constant(value=0))
        self.projection = paddle.nn.Linear(1024, 512, weight_attr=weight_attr, bias_attr=bias_attr)

    @classmethod
    def from_config(cls, cfg, input_shape):
        roi_pooler = cfg['roi_extractor']
        assert isinstance(roi_pooler, dict)
        kwargs = RoIAlign.from_config(cfg, input_shape)
        roi_pooler.update(kwargs)
        kwargs = {'input_shape': input_shape}
        head = create(cfg['head'], **kwargs)
        return {
            'roi_extractor': roi_pooler,
            'head': head,
            'in_channel': head.out_shape[0].channels
        }

    def forward(self, body_feats=None, rois=None, rois_num=None, inputs=None):
        """
        body_feats (list[Tensor]): Feature maps from backbone
        rois (list[Tensor]): RoIs generated from RPN module
        rois_num (Tensor): The number of RoIs in each image
        inputs (dict{Tensor}): The ground-truth of image
        """
        if self.training:
            rois, rois_num, targets = self.bbox_assigner(rois, rois_num, inputs)
            self.assigned_rois = (rois, rois_num)
            self.assigned_targets = targets

        rois_feat = self.roi_extractor(body_feats, rois, rois_num)
        bbox_feat = self.head(rois_feat)
        if self.with_pool:
            feat = F.adaptive_avg_pool2d(bbox_feat, output_size=1)
            feat = paddle.squeeze(feat, axis=[2, 3])
        else:
            feat = bbox_feat
        scores = self.bbox_score(feat)
        deltas = self.bbox_delta(feat)

        if self.training:
            loss = self.get_loss(scores, deltas, targets, rois,
                                 self.bbox_weight)
            return loss, bbox_feat
        else:
            pred = self.get_prediction(scores, deltas)
            return pred, self.head

    def get_loss(self, scores, deltas, targets, rois, bbox_weight):
        """
        scores (Tensor): scores from bbox head outputs
        deltas (Tensor): deltas from bbox head outputs
        targets (list[List[Tensor]]): bbox targets containing tgt_labels, tgt_bboxes and tgt_gt_inds
        rois (List[Tensor]): RoIs generated in each batch
        """
        cls_name = 'loss_bbox_cls'
        reg_name = 'loss_bbox_reg'
        loss_bbox = {}

        # TODO: better pass args
        tgt_labels, tgt_bboxes, tgt_gt_inds = targets

        # bbox cls
        tgt_labels = paddle.concat(tgt_labels) if len(
            tgt_labels) > 1 else tgt_labels[0]
        valid_inds = paddle.nonzero(tgt_labels >= 0).flatten()
        if valid_inds.shape[0] == 0:
            loss_bbox[cls_name] = paddle.zeros([1], dtype='float32')
        else:
            tgt_labels = tgt_labels.cast('int64')
            tgt_labels.stop_gradient = True
            loss_bbox_cls = F.cross_entropy(
                input=scores, label=tgt_labels, reduction='mean')
            loss_bbox[cls_name] = loss_bbox_cls

        # bbox reg

        cls_agnostic_bbox_reg = deltas.shape[1] == 4

        fg_inds = paddle.nonzero(
            paddle.logical_and(tgt_labels >= 0, tgt_labels <
                               self.num_classes)).flatten()

        if fg_inds.numel() == 0:
            loss_bbox[reg_name] = paddle.zeros([1], dtype='float32')
            return loss_bbox

        if cls_agnostic_bbox_reg:
            reg_delta = paddle.gather(deltas, fg_inds)
        else:
            fg_gt_classes = paddle.gather(tgt_labels, fg_inds)

            reg_row_inds = paddle.arange(fg_gt_classes.shape[0]).unsqueeze(1)
            reg_row_inds = paddle.tile(reg_row_inds, [1, 4]).reshape([-1, 1])

            reg_col_inds = 4 * fg_gt_classes.unsqueeze(1) + paddle.arange(4)

            reg_col_inds = reg_col_inds.reshape([-1, 1])
            reg_inds = paddle.concat([reg_row_inds, reg_col_inds], axis=1)

            reg_delta = paddle.gather(deltas, fg_inds)
            reg_delta = paddle.gather_nd(reg_delta, reg_inds).reshape([-1, 4])
        rois = paddle.concat(rois) if len(rois) > 1 else rois[0]
        tgt_bboxes = paddle.concat(tgt_bboxes) if len(
            tgt_bboxes) > 1 else tgt_bboxes[0]

        reg_target = bbox2delta(rois, tgt_bboxes, bbox_weight)
        reg_target = paddle.gather(reg_target, fg_inds)
        reg_target.stop_gradient = True

        if self.bbox_loss is not None:
            reg_delta = self.bbox_transform(reg_delta)
            reg_target = self.bbox_transform(reg_target)
            loss_bbox_reg = self.bbox_loss(
                reg_delta, reg_target).sum() / tgt_labels.shape[0]
            loss_bbox_reg *= self.num_classes
        else:
            loss_bbox_reg = paddle.abs(reg_delta - reg_target).sum(
            ) / tgt_labels.shape[0]

        loss_bbox[reg_name] = loss_bbox_reg

        return loss_bbox

    def bbox_transform(self, deltas, weights=[0.1, 0.1, 0.2, 0.2]):
        wx, wy, ww, wh = weights

        deltas = paddle.reshape(deltas, shape=(0, -1, 4))

        dx = paddle.slice(deltas, axes=[2], starts=[0], ends=[1]) * wx
        dy = paddle.slice(deltas, axes=[2], starts=[1], ends=[2]) * wy
        dw = paddle.slice(deltas, axes=[2], starts=[2], ends=[3]) * ww
        dh = paddle.slice(deltas, axes=[2], starts=[3], ends=[4]) * wh

        dw = paddle.clip(dw, -1.e10, np.log(1000. / 16))
        dh = paddle.clip(dh, -1.e10, np.log(1000. / 16))

        pred_ctr_x = dx
        pred_ctr_y = dy
        pred_w = paddle.exp(dw)
        pred_h = paddle.exp(dh)

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        x1 = paddle.reshape(x1, shape=(-1, ))
        y1 = paddle.reshape(y1, shape=(-1, ))
        x2 = paddle.reshape(x2, shape=(-1, ))
        y2 = paddle.reshape(y2, shape=(-1, ))

        return paddle.concat([x1, y1, x2, y2])

    def get_prediction(self, score, delta):
        bbox_prob = F.softmax(score)
        return delta, bbox_prob

    def get_head(self, ):
        return self.head

    def get_assigned_targets(self, ):
        return self.assigned_targets

    def get_assigned_rois(self, ):
        return self.assigned_rois
    def forward_train(self,
                      x,
                      img,
                      img_no_normalize,
                      img_metas,
                      proposal_list,
                      proposals_pre_computed,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x,img,img_no_normalize,sampling_results,proposals_pre_computed,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses


    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def clip_image_forward_align(self,img,bboxes,num_proposals_per_img,flag=False):
        #==========不确定boxes_num是否为256========
        cropped_images = paddle.vision.ops.roi_align(img,bboxes,output_size=(224,224),boxes_num=256)
        image_features = self.clip_model.model.encode_image(cropped_images)
        return image_features

    def clip_image_forward(self,img_metas,bboxes,num_proposals_per_img,flag=False):
        imgs = []
        bboxes = list(bboxes.clone().split(num_proposals_per_img))
        scale_factors = tuple(img_meta['scale_factor'] for img_meta in img_metas)
        for i,img_meta in enumerate(img_metas):
            img_bytes = self.file_client.get(img_meta['filename'])
            buff = io.BytesIO(img_bytes)
            im = Image.open(buff)
            imgs.append(im)
            if img_meta['flip']:
                w = img_meta['img_shape'][1]
                flipped = bboxes[i].clone()
                flipped[..., 1::4] = w - bboxes[i][..., 3::4]
                flipped[..., 3::4] = w - bboxes[i][..., 1::4]
                bboxes[i] = flipped
        # if self.rank == 0:
        #     print(img_metas[0],imgs[0].size)
        #     print(bboxes[0][:,1:]/bboxes[0].new_tensor(scale_factors[0]),self.proposals[self.file_idxs[img_metas[0]['ori_filename']]][0])
        cropped_images = []
        for img_id,bbox in enumerate(bboxes):
            bbox_raw = bbox[:,1:]
            bbox_raw /= bbox_raw.new_tensor(scale_factors[img_id])
            img_shape = imgs[img_id].size
            # bbox = bbox_raw
            # bbox = torch.dstack([torch.floor(bbox_raw[:,0]),torch.floor(bbox_raw[:,1]),torch.ceil(bbox_raw[:,2]),torch.ceil(bbox_raw[:,3])]).squeeze(0)
            # bbox = torch.stack([torch.floor(bbox_raw[:,0]-0.001),torch.floor(bbox_raw[:,1]-0.001),torch.ceil(bbox_raw[:,2]+0.001),torch.ceil(bbox_raw[:,3]+0.001)]).squeeze(0)
            bbox = paddle.stack([paddle.floor(bbox_raw[:, 0] - 0.001), paddle.floor(bbox_raw[:, 1] - 0.001),
                                paddle.ceil(bbox_raw[:, 2] + 0.001), paddle.ceil(bbox_raw[:, 3] + 0.001)],axis=2).squeeze(0)
            # bbox[:, [0, 2]] = bbox[:,[0,2]].clamp_(min=0,max=img_shape[0])
            bbox[:,0] = paddle.clip(bbox[:,0], min=0, max=img_shape[0])
            # bbox[:, 2] = bbox[:,2].clamp_(min=0, max=img_shape[0])
            bbox[:, 2] = paddle.clip(bbox[:,2], min=0, max=img_shape[0])
            # bbox[:, [1, 3]].clamp_(min=0, max=img_shape[1])
            bbox[:, 1] = paddle.clip(bbox[:, 1], min=0, max=img_shape[1])
            bbox[:, 3] = paddle.clip(bbox[:, 3], min=0, max=img_shape[1])

            bbox = bbox.detach().cpu().numpy()
            # bbox = np.dstack([np.floor(bbox_raw[:,0]),np.floor(bbox_raw[:,1]),np.ceil(bbox_raw[:,2]),np.ceil(bbox_raw[:,3])]).squeeze(0)
            cnt = -1
            for box in bbox:
                cnt += 1
                cropped_image = imgs[img_id].crop(box)
                # if flag:
                    # cropped_image.save('workdirs/output_proposals_15/' + str(cnt) + '_' + img_metas[img_id]['filename'].split('/')[-1])
                try:
                    cropped_image = self.preprocess(cropped_image).to(self.device)
                except:
                    print(img_metas[img_id]['flip'],flag)
                    print(box)
                    raise RuntimeError
                cropped_images.append(cropped_image)
        cropped_images = paddle.stack(cropped_images)
        image_features = self.clip_model.model.encode_image(cropped_images)
        return image_features

    def boxto15(self, bboxes):
        if bboxes.shape[1] == 5:
            bboxes15 = paddle.stack([
                        bboxes[:,0],
                        1.25 * bboxes[:, 1] - 0.25 * bboxes[:, 3],
                        1.25 * bboxes[:, 2] - 0.25 * bboxes[:, 4],
                        1.25 * bboxes[:, 3] - 0.25 * bboxes[:, 1],
                        1.25 * bboxes[:, 4] - 0.25 * bboxes[:, 2]
                        ],axis=2).squeeze(0)
        else:
            bboxes15 = paddle.stack([
                        1.25 * bboxes[:, 0] - 0.25 * bboxes[:, 2],
                        1.25 * bboxes[:, 1] - 0.25 * bboxes[:, 3],
                        1.25 * bboxes[:, 2] - 0.25 * bboxes[:, 0],
                        1.25 * bboxes[:, 3] - 0.25 * bboxes[:, 1]
                        ],axis=2).squeeze(0)
        return bboxes15

    def checkdir(self, path):
        path_prefix = osp.dirname(path)
        if not osp.exists(path_prefix):
            os.makedirs(path_prefix)

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = rois.float()
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        region_embeddings = self.bbox_head.forward_embedding(bbox_feats)
        bbox_pred = self.bbox_head(region_embeddings)
        bbox_results = dict(
            bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results, region_embeddings

    def _bbox_forward_for_image(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = rois.float()
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        region_embeddings = self.bbox_head.forward_embedding_for_image(bbox_feats)

        return None, region_embeddings

    def img2pil2feat(self, img, boxs, name=None):
        img = np.array(img.detach().cpu()).astype(np.uint8)
        img = Image.fromarray(img.transpose(1, 2, 0))
        img_shape = img.size
        # print(img.mode)
        # print(img.size)
        # print(boxs)
        boxs = paddle.stack(
            [paddle.floor(boxs[:, 0] - 0.001), paddle.floor(boxs[:, 1] - 0.001), paddle.ceil(boxs[:, 2] + 0.001),
             paddle.ceil(boxs[:, 3] + 0.001)],axis=2).squeeze(0)
        # boxs = torch.dstack([torch.floor(boxs[:,0]),torch.floor(boxs[:,1]),torch.ceil(boxs[:,2]),torch.ceil(boxs[:,3])]).squeeze(0)
        # boxs[:, [0, 2]].clamp_(min=0, max=img_shape[0])
        # boxs[:, [1, 3]].clamp_(min=0, max=img_shape[1])
        boxs[:, 0] = paddle.clip(boxs[:, 0], min=0, max=img_shape[0])
        # bbox[:, 2] = bbox[:,2].clamp_(min=0, max=img_shape[0])
        boxs[:, 2] = paddle.clip(boxs[:, 2], min=0, max=img_shape[0])
        # bbox[:, [1, 3]].clamp_(min=0, max=img_shape[1])
        boxs[:, 1] = paddle.clip(boxs[:, 1], min=0, max=img_shape[1])
        boxs[:, 3] = paddle.clip(boxs[:, 3], min=0, max=img_shape[1])
        boxs = boxs.detach().cpu().numpy()
        # print(boxs)
        preprocessed = []
        i = 0
        for box in boxs:
            try:
                croped = img.crop(box)
            except:
                print(box)
            # croped.save(name+f'_pil_{i}.jpg')
            i += 1
            croped = self.preprocess(croped)
            preprocessed.append(croped)

        preprocessed = paddle.stack(preprocessed).cuda()
        features = self.clip_model.mdoel.encode_image(preprocessed)
        return features

    def _bbox_forward_train(self, x, img, img_no_normalize, sampling_results, proposals_pre_computed, gt_bboxes,
                            gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        if not self.fix_bg:
            # input_one = x[0].new_ones(1)
            input_one = paddle.ones([1],dtype=x[0].dtype)
            bg_class_embedding = self.bg_embedding(input_one).reshape(1, 512)
            bg_class_embedding = paddle.nn.functional.normalize(bg_class_embedding, p=2, axis=1)
        # ----------------------------------------------------------
        num_proposals_per_img = tuple(len(proposal) for proposal in proposals_pre_computed)
        rois_image = paddle.concat(proposals_pre_computed, axis=0)
        batch_index = paddle.concat([paddle.full(shape=[num_proposals_per_img[i],1],fill_value=i,dtype=x[0].dtype) for i in range(len(num_proposals_per_img))],0)
        rois_image = paddle.concat([batch_index, rois_image[..., :4]], axis=-1)
        bboxes = rois_image
        # bboxes = rois
        # bboxes = bbox2roi(gt_bboxes)
        # ------------------------------------------------------------
        # not using precomputed proposals
        # num_proposals_per_img = tuple(len(gt_bbox) for gt_bbox in gt_bboxes)
        # num_proposals_per_img = tuple(len(res.bboxes) for res in sampling_results)
        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        # if len(gt_bboxes[0])==0:
        # bboxes = rois
        # num_proposals_per_img = tuple(len(res.bboxes) for res in sampling_results)
        # bboxes = rois
        # -------------------------------------------------------------
        if self.ensemble:
            _, region_embeddings_image = self._bbox_forward_for_image(x, bboxes)
            region_embeddings_image = self.projection_for_image(region_embeddings_image)
            region_embeddings_image = paddle.nn.functional.normalize(region_embeddings_image, p=2, axis=1)
        else:
            _, region_embeddings_image = self._bbox_forward(x, bboxes)
            region_embeddings_image = self.projection(region_embeddings_image)
            region_embeddings_image = paddle.nn.functional.normalize(region_embeddings_image, p=2, axis=1)

        if self.load_feature:
            clip_image_features_ensemble = []
            bboxes_all = bboxes.split(num_proposals_per_img)
            #--------
            for i in range(len(img_metas)):
                if self.num_classes == 1203:
                    save_path = os.path.join('lvis_clip_image_embedding.zip/data/lvis_clip_image_embedding',
                                             img_metas[i]['ori_filename'].split('.')[0] + '.pdparams')
                elif self.num_classes == 80:
                    # save_path = os.path.join('lvis_clip_image_embedding.zip/data/lvis_clip_image_embedding/train2017', img_metas[i]['ori_filename'].split('.')[0] + '.pth')
                    save_path = os.path.join('coco_clip_image_embedding.zip/data/coco_clip_image_embedding/',
                                             img_metas[i]['ori_filename'].split('.')[0] + '.pdparams')
                try:
                    f = self.zipfile.get(save_path)
                    stream = io.BytesIO(f)
                    tmp = paddle.load(stream)
                    clip_image_features_ensemble.append(tmp.cuda())
                except:
                    bboxes_single_image = bboxes_all[i]
                    bboxes15 = self.boxto15(bboxes_single_image)
                    if self.num_classes == 1203:
                        save_path = os.path.join('./data/lvis_clip_image_embedding',
                                                 img_metas[i]['ori_filename'].split('.')[0] + '.pdparams')
                    elif self.num_classes == 80:
                        save_path = os.path.join('./data/coco_clip_image_embedding',
                                                 img_metas[i]['ori_filename'].split('.')[0] + '.pdparams')
                    # save_path = osp.join('./data/lvis_clip_image_embedding', img_metas[i]['ori_filename'].split('.')[0] + '.pth')
                    self.checkdir(save_path)
                    # clip_image_features = self.clip_image_forward((img_metas[i],), bboxes[:,1:],(num_proposals_per_img[i],))
                    # clip_image_features15 = self.clip_image_forward((img_metas[i],), bboxes15[:,1:],(num_proposals_per_img[i],))
                    clip_image_features = self.img2pil2feat(img_no_normalize[i], bboxes_single_image[:, 1:])
                    clip_image_features15 = self.img2pil2feat(img_no_normalize[i], bboxes15[:, 1:])
                    clip_image_features_single = clip_image_features + clip_image_features15
                    clip_image_features_single = clip_image_features_single.astype('float32')
                    clip_image_features_single = paddle.nn.functional.normalize(clip_image_features_single, p=2, axis=1)
                    paddle.save(clip_image_features_single.cpu(), save_path)
                    clip_image_features_ensemble.append(clip_image_features_single)
            clip_image_features_ensemble = paddle.concat(clip_image_features_ensemble, axis=0)
        else:
            clip_image_features_ensemble = []
            clip_image_features_ensemble_align = []
            bboxes_all = bboxes.split(num_proposals_per_img)
            for i in range(len(img_metas)):
                bboxes_single_image = bboxes_all[i]
                bboxes15 = self.boxto15(bboxes_single_image)
                if self.num_classes == 1203:
                    save_path = os.path.join('./data/lvis_clip_image_embedding',
                                             img_metas[i]['ori_filename'].split('.')[0] + '.pdparams')
                elif self.num_classes == 80:
                    save_path = os.path.join('./data/coco_clip_image_embedding_ori_forward',
                                             img_metas[i]['ori_filename'].split('.')[0] + '.pdparams')
                self.checkdir(save_path)
                clip_image_features = self.img2pil2feat(img_no_normalize[i], bboxes_single_image[:, 1:])
                clip_image_features15 = self.img2pil2feat(img_no_normalize[i], bboxes15[:, 1:])

                # clip_image_features = self.clip_image_forward((img_metas[i],), bboxes_single_image, (num_proposals_per_img[i],))
                # clip_image_features15 = self.clip_image_forward((img_metas[i],), bboxes15,(num_proposals_per_img[i],),True)

                # clip_image_features_align = self.clip_image_forward_align(img, bboxes,(num_proposals_per_img[i],))
                # clip_image_features15_align = self.clip_image_forward_align(img, bboxes15,(num_proposals_per_img[i],))
                # clip_image_features_single_align = clip_image_features_align + clip_image_features15_align
                # clip_image_features_single_align = clip_image_features_single_align.float()
                # clip_image_features_single_align = torch.nn.functional.normalize(clip_image_features_single_align, p=2, dim=1)
                # clip_image_features_ensemble_align.append(clip_image_features_single_align)

                clip_image_features_single = clip_image_features + clip_image_features15
                clip_image_features_single = clip_image_features_single.astype('float32')
                clip_image_features_single = paddle.nn.functional.normalize(clip_image_features_single, p=2, axis=1)

                clip_image_features_ensemble.append(clip_image_features_single)
                paddle.save(clip_image_features_single.cpu(), save_path)
            clip_image_features_ensemble = paddle.concat(clip_image_features_ensemble, axis=0)
            # clip_image_features_ensemble_align = torch.cat(clip_image_features_ensemble_align, dim=0)
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        labels, _, _, _ = bbox_targets

        region_embeddings = self.projection(region_embeddings)
        region_embeddings = paddle.nn.functional.normalize(region_embeddings, p=2, axis=1)
        if not self.fix_bg:
            text_features = paddle.concat([self.text_features_for_classes, bg_class_embedding], axis=0)
        else:
            text_features = self.text_features_for_classes

        # clip_logits_align = clip_image_features_ensemble_align @ text_features.T
        # clip_logits_align[:,-1] = -1e11
        self.iters += 1
        if self.iters < 200:
            clip_logits = clip_image_features_ensemble @ text_features.T
            clip_logits[:, -1] = -1e11
            num_imgs = len(img_metas)
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
            labels_image = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposals_pre_computed[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                labels_image += assign_result.labels
            labels_image = paddle.to_tensor(labels_image, place=self.device)

            # fg_index = labels_image.ge(0)
            fg_index = paddle.greater_equal(labels_image,0).astype('int32')
            self.accuracy = self.accuracy[-1000:]
            self.accuracy += (clip_logits.argmax(axis=1).equal(labels_image))[fg_index].detach().cpu().tolist()
            print(np.mean(self.accuracy))
        # if self.is_main_process():
        #     print('#'*100)
        #     print(rois[:10,:])
        #     print(clip_logits.argmax(dim=1)[:10],clip_logits_align.argmax(dim=1)[:10])
        # print(fg_index.sum())
        # if len(gt_bboxes[0])>0:
        # self.accuracy_align = self.accuracy_align[-1000:]
        # self.accuracy_align += (clip_logits_align.argmax(dim=1).eq(labels))[fg_index].detach().cpu().tolist()
        # print('align:{} Image crop:{}'.format(np.mean(self.accuracy_align),np.mean(self.accuracy)))

        cls_score_text = region_embeddings @ text_features.T
        # self.iters += 1
        cls_score_text[:, self.novel_label_ids] = -1e11
        # 这里是texthaed的
        text_cls_loss = F.cross_entropy(cls_score_text / self.temperature, labels, reduction='mean')
        # 这里是imagehead的
        kd_loss = F.l1_loss(region_embeddings_image, clip_image_features_ensemble)
        loss_bbox = self.bbox_head.loss(
            bbox_results['bbox_pred'], rois,
            *bbox_targets)
        loss_bbox.update(text_cls_loss=text_cls_loss, kd_loss=kd_loss * self.kd_weight)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    paddle.ones(
                        res.pos_bboxes.shape[0],
                        dtype=paddle.bool).cuda())
                pos_inds.append(
                    paddle.zeros(
                        res.neg_bboxes.shape[0],
                        dtype=paddle.bool).cuda())
            pos_inds = paddle.concat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = paddle.concat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                # proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test_bboxes(self,
                           x,
                           img,
                           img_no_normalize,
                           img_metas,
                           proposals,
                           proposals_pre_computed,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        # get origin input shape to support onnx dynamic input shape
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        # if self.use_clip_inference:
        # proposals = proposals_pre_computed
        rois = bbox2roi(proposals)
        num_proposals_per_img = tuple(len(proposal) for proposal in proposals)
        # rois_image = torch.cat(proposals_pre_computed, dim=0)
        # batch_index = torch.cat([x[0].new_full((num_proposals_per_img[i],1),i) for i in range(len(num_proposals_per_img))],0)
        # rois = torch.cat([batch_index, rois_image[..., :4]], dim=-1)

        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        region_embeddings = self.projection(region_embeddings)
        region_embeddings = paddle.nn.functional.normalize(region_embeddings, p=2, axis=1)
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            bg_class_embedding = paddle.nn.functional.normalize(bg_class_embedding, p=2, axis=1)
            text_features = paddle.concat([self.text_features_for_classes, bg_class_embedding], axis=0)
        else:
            text_features = self.text_features_for_classes
        # -----------------------------------------------------
        # """
        cls_score_text = region_embeddings @ text_features.T

        if self.num_classes == 80 and self.coco_setting:
            cls_score_text[:, self.unseen_label_ids_test] = -1e11
        cls_score_text = cls_score_text / 0.007
        # cls_score_text = cls_score_text/cls_score_text.std(dim=1,keepdim=True)*4
        # cls_score_text = cls_score_text.softmax(dim=1)
        cls_score_text = F.softmax(cls_score_text,axis=1)

        # --------------------------------------------
        if self.ensemble and not self.use_clip_inference:
            # """
            # bbox_pred = bbox_results['bbox_pred']
            # num_proposals_per_img = tuple(len(p) for p in proposals)
            # rois = rois.split(num_proposals_per_img, 0)
            # # some detector with_reg is False, bbox_pred will be None
            # if bbox_pred is not None:
            #     # the bbox prediction of some detectors like SABL is not Tensor
            #     if isinstance(bbox_pred, torch.Tensor):
            #         bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            #     else:
            #         bbox_pred = self.bbox_head.bbox_pred_split(
            #             bbox_pred, num_proposals_per_img)
            # bboxes = []
            # for i in range(len(proposals)):
            #     bbox = self.bbox_head.compute_bboxes(
            #     rois[i],
            #     bbox_pred[i],
            #     img_shapes[i],
            #     scale_factors[i],
            #     rescale=rescale,
            #     cfg=None)
            #     bboxes.append(bbox)
            # bboxes = torch.cat(bboxes,0)
            # """
            # rois_image = bbox2roi(bboxes[:,:4])
            _, region_embeddings_image = self._bbox_forward_for_image(x, rois)
            region_embeddings_image = self.projection_for_image(region_embeddings_image)
            region_embeddings_image = paddle.nn.functional.normalize(region_embeddings_image, p=2, axis=1)
            cls_score_image = region_embeddings_image @ text_features.T
            cls_score_image = cls_score_image / 0.007
            if self.num_classes == 80 and self.coco_setting:
                cls_score_image[:, self.unseen_label_ids_test] = -1e11
            # cls_score_image[:,:-1] = cls_score_image[:,:-1]/cls_score_image[:,:-1].std(dim=1,keepdim=True)*4
            cls_score_image[:, -1] = -1e11
            # cls_score_image = cls_score_image.softmax(dim=1)
            cls_score_image = F.softmax(cls_score_image,axis=1)
        # ------------------------------------------------
        # using clip to inference
        if self.ensemble and self.use_clip_inference:
            bboxes = rois
            save_path = os.path.join('./data/lvis_clip_image_embedding_test_offline',
                                     img_metas[0]['ori_filename'].split('.')[0] + '.pdparams')
            # save_path = os.path.join('./data/lvis_clip_image_embedding_test_offline_img2pil', img_metas[0]['ori_filename'].split('.')[0] + '.pth')
            if not osp.exists(save_path):
                # if True:
                bboxes15 = self.boxto15(bboxes)

                # clip_image_features_img2pil = self.img2pil2feat(img_no_normalize[0], bboxes[:,1:])
                # clip_image_features15_img2pil = self.img2pil2feat(img_no_normalize[0], bboxes15[:,1:])
                # clip_image_features_ensemble_img2pil = clip_image_features_img2pil + clip_image_features15_img2pil
                # clip_image_features_ensemble_img2pil = clip_image_features_ensemble_img2pil.float()
                # clip_image_features_ensemble_img2pil = F.normalize(clip_image_features_ensemble_img2pil,p=2,dim=1)

                # clip_image_features = self.clip_image_forward(img_metas,bboxes,num_proposals_per_img)
                # clip_image_features15 = self.clip_image_forward(img_metas, bboxes15, num_proposals_per_img)
                # clip_image_features_ensemble = clip_image_features + clip_image_features15
                # clip_image_features_ensemble = clip_image_features_ensemble.float()
                # clip_image_features_ensemble = F.normalize(clip_image_features_ensemble,p=2,dim=1)

                clip_image_features_align = self.clip_image_forward_align(img, bboxes, num_proposals_per_img)
                clip_image_features15_align = self.clip_image_forward_align(img, bboxes15, num_proposals_per_img)
                clip_image_features_ensemble_align = clip_image_features_align + clip_image_features15_align
                clip_image_features_ensemble_align = clip_image_features_ensemble_align.astype('float32')
                clip_image_features_ensemble = F.normalize(clip_image_features_ensemble_align, p=2, axis=1)

                # torch.save(clip_image_features_ensemble_img2pil.cpu(), save_path)
                self.checkdir(save_path)
                # torch.save(clip_image_features_ensemble.cpu(), save_path)
            else:
                clip_image_features_ensemble = paddle.load(save_path).cuda()
                # clip_image_features_ensemble_img2pil = torch.load(save_path).to(self.device)
            # cls_score_clip[:,:-1] = cls_score_clip[:,:-1]/cls_score_clip[:,:-1].std(dim=1,keepdim=True)*0.006
            # print(cls_score_clip.std(dim=1).mean())
            cls_score_clip = clip_image_features_ensemble @ text_features.T
            cls_score_clip[:, :-1] = cls_score_clip[:, :-1] / cls_score_clip[:, :-1].std(axis=1, keepdim=True) * 4
            # cls_score_clip = torch.exp(cls_score_clip-1)
            # cls_score_clip = cls_score_clip/0.007
            cls_score_clip[:, -1] = -1e11
            if self.num_classes == 80 and self.coco_setting:
                cls_score_clip[:, self.unseen_label_ids_test] = -1e11
            # cls_score_clip = cls_score_clip.softmax(dim=1)
            cls_score_clip = F.softmax(cls_score_clip,axis=1)
            # cls_score_clip_img2pil = clip_image_features_ensemble_img2pil @ text_features.T
            # cls_score_clip_img2pil = torch.exp(cls_score_clip_img2pil-1)
            # cls_score_clip_img2pil = cls_score_clip_img2pil/0.007
            # cls_score_clip_img2pil[:,-1] = -1e11
            # cls_score_clip_img2pil = cls_score_clip_img2pil.softmax(dim=1)

            # cls_score_clip_align = clip_image_features_ensemble_align @ text_features.T
            # cls_score_clip_align = torch.exp(cls_score_clip_align-1)
            # cls_score_clip_align = cls_score_clip_align/0.007
            # cls_score_clip_align[:,-1] = -1e11
            # cls_score_clip_align = cls_score_clip_align.softmax(dim=1)
            cls_score_image = cls_score_clip
        # --------------------------------------------------
        # """
        a = 1 / 3
        if self.ensemble:
            if self.fixed_lambda is not None:
                cls_score = cls_score_image ** (1 - self.fixed_lambda) * cls_score_text ** self.fixed_lambda
            else:
                cls_score = paddle.where(self.novel_index, cls_score_image ** (1 - a) * cls_score_text ** a,
                                        cls_score_text ** (1 - a) * cls_score_image ** a)
                # print(11)

            # cls_score_align= torch.where(self.novel_index,cls_score_clip_align**(1-a)*cls_score_text**a,
            #    cls_score_text**(1-a)*cls_score_clip_align**a)
            # cls_score = cls_score_image**(1-a)*cls_score_text**a
            # cls_score = cls_score_image
        else:
            cls_score = cls_score_text
        # """
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, paddle.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            # if self.use_clip_inference:
            #     # proposal_label = self.novel_label_ids[cls_score[i][:,self.novel_label_ids].argmax(dim=1)]
            # for j,label in enumerate(proposal_label):
            #     box = proposals[i][j].detach().cpu().numpy().tolist()
            #     print('{} {} {} {} {} {}'.format(img_metas[0]['ori_filename'],cls_score[i].max(dim=1)[0][j],box[0],box[1],box[2],box[3]),file=open('/home/dy20/mmdetection27/workdirs/det_result/train_novel_det/{}_det_{}.txt'.format(self.rank,label),'a'))
        return det_bboxes, det_labels

    def simple_test(self,
                    x,
                    img,
                    img_no_normalize,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x,img,img_no_normalize, img_metas, proposal_list,proposals, self.test_cfg, rescale=rescale)
        # if torch.onnx.is_in_onnx_export():
        #     if self.with_mask:
        #         segm_results = self.simple_test_mask(
        #             x, img_metas, det_bboxes, det_labels, rescale=rescale)
        #         return det_bboxes, det_labels, segm_results
        #     else:
        #         return det_bboxes, det_labels

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False,**kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            rois = bbox2roi([proposals])
            bbox_results,region_embeddings = self._bbox_forward(x,rois)
            region_embeddings = self.projection(region_embeddings)
            region_embeddings = paddle.nn.functional.normalize(region_embeddings,p=2,axis=1)
            #----
            # input_one = x[0].new_ones(1)
            input_one = new_ones(1,x[0])
            bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            bg_class_embedding = paddle.nn.functional.normalize(bg_class_embedding,p=2,axis=1)
            text_features = paddle.concat([self.text_features_for_classes,bg_class_embedding],axis=0)
            cls_score_text = region_embeddings@text_features.T
            cls_score_text = cls_score_text/0.007
            #0.009#0.008#0.007
            # cls_score_text = cls_score_text.softmax(dim=1)
            cls_score_text = F.softmax(cls_score_text,axis=1)
            cls_score = cls_score_text
            bboxes, scores = self.bbox_head.get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        #------使用paddledetction里面的
        det_results = multiclass_nms(merged_bboxes, merged_scores,
                                               score_threshold=self.test_nms_score_thr,
                                                nms_top_k=self.test_nms_max_per_img,
                                                nms_threshold=self.test_nms_iou_threshold)
        det_bboxes = det_results[:,1:6]
        det_labels = det_results[:,0]
        return det_bboxes, det_labels