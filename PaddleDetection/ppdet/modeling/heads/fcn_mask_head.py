import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from mmcv.cnn import Conv2d, ConvModule, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import auto_fp16, force_fp32


from .file.roiheads.loss import CrossEntropyLoss

from PaddleDetection.ppdet.core.workspace import register
from ...modeling.layers import ConvTranspose2d
from .file.roiheads.utils import new_tensor,arange,new_zeros
import pycocotools.mask as maskUtils

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit

def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg_mask_size):
    """Compute mask target for positive proposals in multiple images.

    Args:
        pos_proposals_list (list[Tensor]): Positive proposals in multiple
            images.
        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each
            positive proposals.
        gt_masks_list (list[:obj:`BaseInstanceMasks`]): Ground truth masks of
            each image.
        cfg (dict): Config dict that specifies the mask size.

    Returns:
        list[Tensor]: Mask target of each image.
    """
    cfg_list = [cfg_mask_size for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = list(mask_targets)
    if len(mask_targets) > 0:
        mask_targets = paddle.concat(mask_targets)
    return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg_mask_size):
    """Compute mask target for each positive proposal in the image.

    Args:
        pos_proposals (Tensor): Positive proposals.
        pos_assigned_gt_inds (Tensor): Assigned GT inds of positive proposals.
        gt_masks (:obj:`BaseInstanceMasks`): GT masks in the format of Bitmap
            or Polygon.
        cfg (dict): Config dict that indicate the mask size.

    Returns:
        Tensor: Mask target of each positive proposals in the image.
    """
    device = pos_proposals.device
    mask_size = (cfg_mask_size,cfg_mask_size)
    num_pos = pos_proposals.shape[0]
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        maxh, maxw = gt_masks.height, gt_masks.width
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

        mask_targets = gt_masks.crop_and_resize(
            proposals_np, mask_size, device=device,
            inds=pos_assigned_gt_inds).to_ndarray()

        # mask_targets = torch.from_numpy(mask_targets).float().to(device)
        mask_targets = paddle.to_tensor(mask_targets,place=device,dtype=paddle.float32)
    else:
        # mask_targets = pos_proposals.new_zeros((0, ) + mask_size)
        mask_targets = new_zeros(shape=(0, ) + mask_size,src=pos_proposals)
    return mask_targets


@register
class FCNMaskHead(nn.Layer):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=80,
                 class_agnostic=False,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=CrossEntropyLoss(use_mask=True,loss_weight=1.0)
                 # dict(
                 #     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
                 ):
        super(FCNMaskHead, self).__init__()
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
                None, 'deconv'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv"')
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = (roi_feat_size,roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_mask = loss_mask

        self.convs = nn.LayerList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            # self.convs.append(
            #     ConvModule(
            #         in_channels,
            #         self.conv_out_channels,
            #         self.conv_kernel_size,
            #         padding=padding,
            #         conv_cfg=conv_cfg,
            #         norm_cfg=norm_cfg))
            self.convs.append(
                nn.Conv2D(in_channels=in_channels,
                          out_channels=self.conv_out_channels,
                          kernel_size=self.conv_kernel_size,
                          padding=padding
                          )
            )
            self.convs.append(nn.ReLU())
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            # upsample_cfg_.update(
            #     in_channels=upsample_in_channels,
            #     out_channels=self.conv_out_channels,
            #     kernel_size=self.scale_factor,
            #     stride=self.scale_factor)
            # self.upsample = build_upsample_layer(upsample_cfg_)

            self.upsample = ConvTranspose2d(in_channels=upsample_in_channels,
                                            out_channels=self.conv_out_channels,
                                            kernel_size=self.scale_factor,
                                            stride=self.scale_factor,
                                            weight_init=nn.initializer.KaimingNormal(),
                                            bias_init=paddle.nn.initializer.Constant(value=0)
                                            )


        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        weight_attr = paddle.framework.ParamAttr(
            name="weight",
            initializer=nn.initializer.KaimingNormal())
        # nn.init.constant_(self.fc_reg.bias, 0)
        bias_attr = paddle.ParamAttr(
            name="bias",
            initializer=paddle.nn.initializer.Constant(value=0))
        self.conv_logits = nn.Conv2D(logits_in_channel, out_channels, 1,weight_attr=weight_attr,bias_attr=bias_attr)
        self.relu = nn.ReLU()
        self.debug_imgs = None

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            elif isinstance(m, CARAFEPack):
                m.init_weights()
            else:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    # @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        if mask_pred.shape[0] == 0:
            loss_mask = mask_pred.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_pred, mask_targets,
                                           paddle.zeros_like(labels))
            else:
                loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, test_config_mask_thr_binary,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, paddle.Tensor):
            mask_pred = F.sigmoid(mask_pred)
        else:
            # mask_pred = det_bboxes.new_tensor(mask_pred)
            mask_pred = new_tensor(mask_pred,det_bboxes)

        device = mask_pred.place
        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            if isinstance(scale_factor, float):
                img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
                img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            else:
                w_scale, h_scale = scale_factor[0], scale_factor[1]
                img_h = np.round(ori_shape[0] * h_scale.item()).astype(
                    np.int32)
                img_w = np.round(ori_shape[1] * w_scale.item()).astype(
                    np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, paddle.Tensor)):
            # scale_factor = bboxes.new_tensor(scale_factor)
            scale_factor = new_tensor(scale_factor,bboxes)
        bboxes = bboxes / scale_factor

        # if torch.onnx.is_in_onnx_export():
        #     # TODO: Remove after F.grid_sample is supported.
        #     from torchvision.models.detection.roi_heads \
        #         import paste_masks_in_image
        #     masks = paste_masks_in_image(mask_pred, bboxes, ori_shape[:2])
        #     thr = rcnn_test_cfg.get('mask_thr_binary', 0)
        #     if thr > 0:
        #         masks = masks >= thr
        #     return masks

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if str(device) == 'Place(cpu)':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        # chunks = torch.chunk(paddle.arange(N, device=device), num_chunks)
        chunks = paddle.chunk(arange(N,place=device),num_chunks)

        threshold = test_config_mask_thr_binary
        # im_mask = paddle.zeros(
        #     [N,img_h,img_w]
        #     device=device,
        #     dtype=torch.bool if threshold >= 0 else torch.uint8)
        im_mask = paddle.to_tensor(np.zeros(shape=[N,img_h,img_w]),dtype=paddle.bool if threshold >= 0 else paddle.uint8)

        if not self.class_agnostic:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=str(device) == 'Place(cpu)')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).cast(dtype=paddle.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).cast(dtype=paddle.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].detach().cpu().numpy())
        return cls_segms


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks acoording to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.place
    if skip_empty:
        x0_int, y0_int = paddle.clip(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).cast(dtype=paddle.int32)
        x1_int = paddle.clip(
            boxes[:, 2].max().ceil() + 1, max=img_w).cast(dtype=paddle.int32)
        y1_int = paddle.clip(
            boxes[:, 3].max().ceil() + 1, max=img_h).cast(dtype=paddle.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = paddle.split(boxes, 1, axis=1)  # each is Nx1

    N = masks.shape[0]

    # img_y = paddle.arange(
    #     y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_y = arange(y0_int,y1_int,place=device,dtype=paddle.float32)+0.5
    # img_x = paddle.arange(
    #     x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_x = arange(x0_int,x1_int,place=device,dtype=paddle.float32)+0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    if paddle.isinf(img_x).any():
        inds = paddle.where(paddle.isinf(img_x))
        img_x[inds] = 0
    if paddle.isinf(img_y).any():
        inds = paddle.where(paddle.isinf(img_y))
        img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.shape[1], img_x.shape[1])
    gy = img_y[:, :, None].expand(N, img_y.shape[1], img_x.shape[1])
    grid = paddle.stack([gx, gy], axis=3)

    # if torch.onnx.is_in_onnx_export():
    #     raise RuntimeError(
    #         'Exporting F.grid_sample from Pytorch to ONNX is not supported.')
    img_masks = F.grid_sample(
        masks.cast(dtype=paddle.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
