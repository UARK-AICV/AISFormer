import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.layers import interpolate, get_instances_contour_interior
from pytorch_toolbelt import losses as L

from pytorch_toolbelt.modules import AddCoords
from detectron2.layers.dice_loss import DiceBCELoss
from detectron2.layers.transformer import *

import copy
from typing import Optional, List
from torch import Tensor
from detectron2.utils.misc import get_masks_from_tensor_list, nested_tensor_from_tensor_list, NestedTensor
from detectron2.layers.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from detectron2.layers.maskencoding import DctMaskEncoding
from detectron2.layers.mlp import MLP
from detectron2.modeling.roi_heads.aisformer.aisformer import AISFormer

ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

# """
import os
import matplotlib.pyplot as plt

def visualize_featmap(featmap, folder_name):
    path_name = os.path.join('../data/outtest/', folder_name)
    os.makedirs(path_name, exist_ok=True)
    np_featmap = featmap.detach().cpu().numpy()
    for i, fm in enumerate(np_featmap):
        plt.imsave(os.path.join(path_name, 'sample_{}.png'.format(str(i))),fm.mean(axis=0))

def visualize_prediction_logits(pred_logits, folder_name):
    path_name = os.path.join('../data/outtest/', folder_name)
    os.makedirs(path_name, exist_ok=True)
    np_pred_sm = pred_logits.sigmoid().detach().cpu().numpy()
    for i, logit_sm in enumerate(np_pred_sm):
        logit_mask = logit_sm > 0.5
        # print(os.path.join(path_name, 'sample_{}.png'.format(str(i))))
        plt.imsave(os.path.join(path_name, 'sample_{}.png'.format(str(i))), logit_mask)

def visualize_gt(gts, folder_name):
    path_name = os.path.join('../data/outtest/', folder_name)
    os.makedirs(path_name, exist_ok=True)

    np_gts = gts.detach().cpu().numpy()
    for i, gt in enumerate(np_gts):
        plt.imsave(os.path.join(path_name, 'sample_{}.png'.format(str(i))), gt)
# """

def mask_rcnn_loss(pred_mask_logits, pred_boundary_logits, instances, pred_mask_bo_logits, pred_boundary_logits_bo, 
                    use_i_mask=False, pred_a_mask_logits=None, pred_a_boundary_logits=None, use_justify_loss=False, c_iter=None, dice_loss=False,
                    pred_invisible_mask_logits=None):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    gt_bo_masks = []
    gt_boundary_bo = []
    gt_boundary = []
    gt_i_masks = []
    gt_i_boundary = []


    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        #print('mask_head.py L59 instances_per_image.gt_masks:', instances_per_image.gt_masks)
        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

        if use_i_mask:
            gt_i_masks_per_image = instances_per_image.gt_i_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_i_masks.append(gt_i_masks_per_image)

        boundary_ls = []
        for mask in gt_masks_per_image:
            mask_b = mask.data.cpu().numpy()
            boundary, inside_mask, weight = get_instances_contour_interior(mask_b)
            boundary = torch.from_numpy(boundary).to(device=mask.device).unsqueeze(0)

            boundary_ls.append(boundary)

        gt_boundary.append(cat(boundary_ls, dim=0))

        if use_i_mask:
            i_boundary_ls = []
            for mask in gt_i_masks_per_image:
                mask_b = mask.data.cpu().numpy()
                boundary, inside_mask, weight = get_instances_contour_interior(mask_b)
                boundary = torch.from_numpy(boundary).to(device=mask.device).unsqueeze(0)

                i_boundary_ls.append(boundary)

            gt_i_boundary.append(cat(i_boundary_ls, dim=0))


        gt_bo_masks_per_image = instances_per_image.gt_bo_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_bo_masks.append(gt_bo_masks_per_image)

        boundary_ls_bo = []
        for mask_bo in gt_bo_masks_per_image:
            mask_b_bo = mask_bo.data.cpu().numpy()
            boundary_bo, inside_mask_bo, weight_bo = get_instances_contour_interior(mask_b_bo)
            boundary_bo = torch.from_numpy(boundary_bo).to(device=mask_bo.device).unsqueeze(0)

            boundary_ls_bo.append(boundary_bo)

        gt_boundary_bo.append(cat(boundary_ls_bo, dim=0))


    '''
    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0,  pred_boundary_logits.sum() * 0
    '''

    if use_i_mask:
        if len(gt_i_masks) == 0:
            return pred_mask_logits.sum() * 0,  pred_boundary_logits.sum() * 0,\
                    pred_mask_logits.sum() * 0,  pred_boundary_logits.sum() * 0,\
                    pred_mask_logits.sum() * 0,  pred_boundary_logits.sum() * 0,\
                    pred_boundary_logits.sum() * 0
    else:
        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0,  pred_boundary_logits.sum() * 0,\
                    pred_mask_logits.sum() * 0,  pred_boundary_logits.sum() * 0,\
                    pred_mask_logits.sum() * 0,  pred_boundary_logits.sum() * 0,\
                    pred_boundary_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)
    gt_bo_masks = cat(gt_bo_masks, dim=0)

    gt_boundary_bo = cat(gt_boundary_bo, dim=0)
    gt_boundary = cat(gt_boundary, dim=0)

    if use_i_mask:
        gt_i_masks = cat(gt_i_masks, dim=0)
        gt_i_boundary = cat(gt_i_boundary, dim=0)
    
    if cls_agnostic_mask:
        pred_mask_logits_gt = pred_mask_logits[:, 0]
        pred_bo_mask_logits = pred_mask_bo_logits[:, 0]
        pred_boundary_logits_bo = pred_boundary_logits_bo[:, 0]
        pred_boundary_logits = pred_boundary_logits[:, 0]
        pred_a_boundary_logits = pred_a_boundary_logits[:, 0]
        if pred_a_mask_logits != None:
            pred_a_mask_logits_gt = pred_a_mask_logits[:, 0]
        if pred_invisible_mask_logits != None:
            pred_invisible_mask_logits_gt = pred_invisible_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits_gt = pred_mask_logits[indices, gt_classes]
        pred_bo_mask_logits = pred_mask_bo_logits[indices, gt_classes]
        pred_boundary_logits_bo = pred_boundary_logits_bo[indices, gt_classes]
        pred_boundary_logits = pred_boundary_logits[indices, gt_classes]
        pred_a_boundary_logits = pred_a_boundary_logits[indices, gt_classes]
        if pred_a_mask_logits != None:
            pred_a_mask_logits_gt = pred_a_mask_logits[:, 0]
        if pred_invisible_mask_logits != None:
            pred_invisible_mask_logits_gt = pred_invisible_mask_logits[:, 0]

    
    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        gt_masks_bool = gt_masks > 0.5

    if use_i_mask:
        if gt_i_masks.dtype == torch.bool:
            gt_i_masks_bool = gt_i_masks
        else:
            gt_i_masks_bool = gt_i_masks > 0.5

    mask_incorrect = (pred_mask_logits_gt > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    if use_i_mask:
        mask_incorrect = (pred_mask_logits_gt > 0.0) != gt_i_masks_bool
        mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
        num_positive = gt_i_masks_bool.sum().item()
        false_positive = (mask_incorrect & ~gt_i_masks_bool).sum().item() / max(
            gt_i_masks_bool.numel() - num_positive, 1.0
        )
        false_negative = (mask_incorrect & gt_i_masks_bool).sum().item() / max(num_positive, 1.0)

    indexs2 = torch.nonzero(torch.sum(gt_bo_masks.to(dtype=torch.float32),(1,2)))

    new_gt_bo_masks1 = gt_bo_masks[indexs2,:,:].squeeze()
    new_gt_bo_masks2 = gt_bo_masks[:indexs2.shape[0]]
    if new_gt_bo_masks1.shape != new_gt_bo_masks2.shape:
        new_gt_bo_masks1 = new_gt_bo_masks1.unsqueeze(0)

    new_gt_bo_masks = torch.cat((new_gt_bo_masks1, new_gt_bo_masks2),0)
    
    pred_bo_mask_logits1 = pred_bo_mask_logits[indexs2,:,:].squeeze()
    pred_bo_mask_logits2 = pred_bo_mask_logits[:indexs2.shape[0]]
    if pred_bo_mask_logits1.shape != pred_bo_mask_logits2.shape:
        pred_bo_mask_logits1 = pred_bo_mask_logits1.unsqueeze(0)

    new_pred_bo_mask_logits = torch.cat((pred_bo_mask_logits1, pred_bo_mask_logits2),0)
 
    new_gt_bo_bounds1 = gt_boundary_bo[indexs2,:,:].squeeze()
    new_gt_bo_bounds2 = gt_boundary_bo[:indexs2.shape[0]]
    if new_gt_bo_bounds1.shape != new_gt_bo_bounds2.shape:
        new_gt_bo_bounds1 = new_gt_bo_bounds1.unsqueeze(0)

    new_gt_bo_bounds = torch.cat((new_gt_bo_bounds1, new_gt_bo_bounds2),0)
    
    pred_bo_bounds_logits1 = pred_boundary_logits_bo[indexs2,:,:].squeeze()
    pred_bo_bounds_logits2 = pred_boundary_logits_bo[:indexs2.shape[0]]
    if pred_bo_bounds_logits1.shape != pred_bo_bounds_logits2.shape:
        pred_bo_bounds_logits1 = pred_bo_bounds_logits1.unsqueeze(0)

    new_pred_bo_bounds_logits = torch.cat((pred_bo_bounds_logits1, pred_bo_bounds_logits2),0)

    # Losses               
    a_mask_loss = torch.tensor(0., requires_grad=True)
    dice_loss_fn = DiceBCELoss()

    '''
    if c_iter % 1000 == 0:
        visualize_prediction_logits(pred_a_mask_logits_gt, 'pred_a_mask_logits_' + str(c_iter))
        visualize_prediction_logits(new_pred_bo_mask_logits, 'pred_bo_mask_logits_' + str(c_iter))
        visualize_prediction_logits(pred_mask_logits_gt, 'pred_i_mask_logits' + str(c_iter))

        visualize_gt(gt_masks, 'gt_a_mask_' + str(c_iter))
        visualize_gt(new_gt_bo_masks, 'gt_bo_mask_' + str(c_iter))
        visualize_gt(gt_i_masks, 'gt_i_mask_' + str(c_iter))
    '''

    if use_i_mask:
        if dice_loss:
            mask_loss = dice_loss_fn(pred_mask_logits_gt, gt_i_masks.to(dtype=torch.float32))
        else:
            mask_loss = F.binary_cross_entropy_with_logits(
                pred_mask_logits_gt, gt_i_masks.to(dtype=torch.float32), reduction="mean"
            )
        if pred_a_mask_logits != None and c_iter >= 0:
            if dice_loss:
                a_mask_loss = dice_loss_fn(pred_a_mask_logits_gt, gt_masks.to(dtype=torch.float32))
            else:
                a_mask_loss = F.binary_cross_entropy_with_logits(
                    pred_a_mask_logits_gt, gt_masks.to(dtype=torch.float32), reduction="mean"
                )
    else:
        if dice_loss:
            mask_loss = dice_loss_fn(pred_mask_logits_gt, gt_masks.to(dtype=torch.float32))
        else:
            mask_loss = F.binary_cross_entropy_with_logits(
                pred_mask_logits_gt, gt_masks.to(dtype=torch.float32), reduction="mean"
            )

    if use_i_mask:
        bound_loss = L.JointLoss(L.BalancedBCEWithLogitsLoss(), L.BalancedBCEWithLogitsLoss())(
            pred_boundary_logits.unsqueeze(1), gt_i_boundary.to(dtype=torch.float32))
        
        a_bound_loss = L.JointLoss(L.BalancedBCEWithLogitsLoss(), L.BalancedBCEWithLogitsLoss())(
            pred_a_boundary_logits.unsqueeze(1), gt_boundary.to(dtype=torch.float32))
    else:
        bound_loss = L.JointLoss(L.BalancedBCEWithLogitsLoss(), L.BalancedBCEWithLogitsLoss())(
            pred_boundary_logits.unsqueeze(1), gt_boundary.to(dtype=torch.float32))
        a_bound_loss = torch.tensor(0.)

    if new_gt_bo_masks.shape[0] > 0: 
        if dice_loss:
            bo_mask_loss = dice_loss_fn(
                    new_pred_bo_mask_logits, new_gt_bo_masks.to(dtype=torch.float32)
                )
        else:
            bo_mask_loss = F.binary_cross_entropy_with_logits(
                new_pred_bo_mask_logits, new_gt_bo_masks.to(dtype=torch.float32), reduction="mean"
            )
    else:
        bo_mask_loss = torch.tensor(0.0).cuda(mask_loss.get_device())

    if new_gt_bo_bounds.shape[0] > 0: 
        bo_bound_loss = L.JointLoss(L.BalancedBCEWithLogitsLoss(), L.BalancedBCEWithLogitsLoss())(
            new_pred_bo_bounds_logits.unsqueeze(1), new_gt_bo_bounds.to(dtype=torch.float32))
    else:
        bo_bound_loss = torch.tensor(0.0).cuda(mask_loss.get_device())

    rec_loss = torch.tensor(0.)
    if use_justify_loss:
        invisible_part_gt = gt_masks ^ gt_i_masks
        rec_loss =  F.binary_cross_entropy_with_logits(
                    pred_invisible_mask_logits_gt, invisible_part_gt.to(dtype=torch.float32), reduction="mean"
                )

    return mask_loss, bo_mask_loss, bound_loss, bo_bound_loss, a_mask_loss, a_bound_loss, rec_loss



def mask_rcnn_inference(pred_mask_logits, bo_mask_logits, bound_logits, bo_bound_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    #pred_mask_logits = pred_mask_logits[:,0:1]
    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
        bound_probs_pred = bound_logits.sigmoid()
        bo_mask_probs_pred = bo_mask_logits.sigmoid()
        bo_bound_probs_pred = bo_bound_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
        bound_probs_pred = bound_logits.sigmoid()
        bo_mask_probs_pred = bo_mask_logits.sigmoid()
        bo_bound_probs_pred = bo_bound_logits.sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
    bo_mask_probs_pred = bo_mask_probs_pred.split(num_boxes_per_image, dim=0)
    bo_bound_probs_pred = bo_bound_probs_pred.split(num_boxes_per_image, dim=0)
    bound_probs_pred = bound_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)
        instances.raw_masks = prob

    for bo_prob, instances in zip(bo_mask_probs_pred, pred_instances):
        instances.pred_masks_bo = bo_prob  # (1, Hmask, Wmask)

    for bo_bound_prob, instances in zip(bo_bound_probs_pred, pred_instances):
        instances.pred_bounds_bo = bo_bound_prob  # (1, Hmask, Wmask)

    for bound_prob, instances in zip(bound_probs_pred, pred_instances):
        instances.pred_bounds = bound_prob  # (1, Hmask, Wmask)

def mask_rcnn_inference_amodal(pred_mask_logits, pred_a_mask_logits, 
                                occluder_mask_logits, invisible_mask_logits, pred_instances):
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    #pred_mask_logits = pred_mask_logits[:,0:1]
    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
        a_mask_probs_pred = pred_a_mask_logits.sigmoid()
        occluder_mask_probs_pred = occluder_mask_logits.sigmoid()
        invisible_mask_probs_pred = invisible_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
        a_mask_probs_pred = pred_a_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
    a_mask_probs_pred = a_mask_probs_pred.split(num_boxes_per_image, dim=0)
    occluder_mask_probs_pred = occluder_mask_probs_pred.split(num_boxes_per_image, dim=0)
    invisible_mask_probs_pred = invisible_mask_probs_pred.split(num_boxes_per_image, dim=0)


    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_visible_masks = prob

    for prob, instances in zip(a_mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)
        instances.raw_masks = prob
        instances.pred_amodal_masks = prob  # (1, Hmask, Wmask)

    for prob, instances in zip(occluder_mask_probs_pred, pred_instances):
        instances.pred_occluder_masks = prob  # (1, Hmask, Wmask)

    for prob, instances in zip(invisible_mask_probs_pred, pred_instances):
        instances.pred_invisible_masks = prob  # (1, Hmask, Wmask)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(MaskRCNNConvUpsampleHead, self).__init__()

        ###############
        if cfg.MODEL.ROI_MASK_HEAD.CUSTOM_NAME == 'AISFormer':
            self.mask_head_model = AISFormer(cfg, input_shape)
            print("AISFormer param: ", count_parameters(self.mask_head_model))
        else:
            assert False, "Invalid custom name for mask head"
        ###############

    def forward(self,x,c_iter,instances=None): # here
        #from fvcore.nn import FlopCountAnalysis
        #flops = FlopCountAnalysis(self.mask_head_model, x)
        #print(flops.total())
        return self.mask_head_model(x)


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)
