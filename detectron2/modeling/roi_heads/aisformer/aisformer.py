
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


import os
import matplotlib.pyplot as plt



class AISFormer(nn.Module):
    def __init__(self,cfg,input_shape: ShapeSpec):
        super(AISFormer, self).__init__()
        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.num_mask_classes = num_mask_classes
        self.aisformer = cfg.MODEL.AISFormer
        # fmt: on

        # deconv
        self.deconv_for_TR = nn.Sequential(
                nn.ConvTranspose2d(conv_dims, conv_dims, kernel_size=(2, 2), stride=(2, 2)),
                nn.ReLU()
            )
        weight_init.c2_msra_fill(self.deconv_for_TR[0])

        # feature map fcn
        self.mask_feat_learner_TR = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), activation=nn.ReLU())
        weight_init.c2_msra_fill(self.mask_feat_learner_TR)

        # mask predictor
        self.predictor_TR = Conv2d(1, num_mask_classes, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.predictor_TR.weight, std=0.001)
        if self.predictor_TR.bias is not None:
            nn.init.constant_(self.predictor_TR.bias, 0)

        # pixel embedding
        self.pixel_embed = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        nn.init.normal_(self.pixel_embed.weight, std=0.001)
        if self.pixel_embed.bias is not None:
            nn.init.constant_(self.pixel_embed.bias, 0)

        # mask embedding
        self.mask_embed = MLP(256,256,256,3)
        for layer in self.mask_embed.layers:
            torch.nn.init.xavier_uniform_(layer.weight)

        # subtract modeling
        self.subtract_model = MLP(512, 256, 256, 2)
        for layer in self.subtract_model.layers:
            torch.nn.init.xavier_uniform_(layer.weight)

        # norm rois
        self.norm_rois = nn.LayerNorm(256)


        # transformer layers
        emb_dim = conv_dims
        self.positional_encoding = PositionEmbeddingLearned(emb_dim//2)

        encoder_layer = TransformerEncoderLayer(d_model=emb_dim, nhead=self.aisformer.N_HEADS, normalize_before=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.aisformer.N_LAYERS)

        decoder_layer = TransformerDecoderLayer(d_model=emb_dim, nhead=self.aisformer.N_HEADS, normalize_before=False)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=self.aisformer.N_LAYERS) # 6 is the default of detr

        n_output_masks = 4 # 4 embeddings, vi_mask, occluder, a_mask, invisible_mask
        self.query_embed = nn.Embedding(num_embeddings=n_output_masks, embedding_dim=emb_dim)


    def forward(self, x):
        x_ori = x.clone()
        bs = x_ori.shape[0]
        emb_dim = x_ori.shape[1]
        spat_size = x_ori.shape[-1]

        # short range learning
        x = self.mask_feat_learner_TR(x)
        x = self.deconv_for_TR(x)

        # position emb
        pos_embed = self.positional_encoding.forward_tensor(x)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        # encode
        feat_embs = x.flatten(2).permute(2, 0, 1)
        encoded_feat_embs = self.transformer_encoder(feat_embs, 
                                                    pos=pos_embed)

        # decode
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed) 
        decoder_output = self.transformer_decoder(tgt, encoded_feat_embs, 
                                        pos=pos_embed, 
                                        query_pos=query_embed) # (1, n_masks, bs, dim)

        decoder_output = decoder_output.squeeze(0).moveaxis(1,0)

        # predict mask
        roi_embeding =  encoded_feat_embs.permute(1,2,0).unflatten(-1, (28,28))
        roi_embeding = roi_embeding + x # long range + short range
        roi_embeding = self.norm_rois(roi_embeding.permute(0,2,3,1)).permute(0,3,1,2)
        roi_embeding = self.pixel_embed(roi_embeding)

        mask_embs = self.mask_embed(decoder_output)
        if self.aisformer.JUSTIFY_LOSS:
            assert self.aisformer.USE == True
            combined_feat = torch.cat([mask_embs[:,2,:],mask_embs[:,0,:]], axis=1)
            invisible_embs = self.subtract_model(combined_feat)
            invisible_embs = invisible_embs.unsqueeze(1)
            mask_embs = torch.concat([mask_embs, invisible_embs], axis=1)

        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embs, roi_embeding)

        vi_masks        = outputs_mask[:,0,:,:].unsqueeze(1) #visible mask
        bo_masks        = outputs_mask[:,1,:,:].unsqueeze(1) #occluder mask
        a_masks         = outputs_mask[:,2,:,:].unsqueeze(1) #amodal mask
        invisible_masks = outputs_mask[:,-1,:,:].unsqueeze(1) #invisible mask
        dump_tensor = torch.zeros_like(vi_masks).to(device='cuda')

        return vi_masks, bo_masks, a_masks, invisible_masks
