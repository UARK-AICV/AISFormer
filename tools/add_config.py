from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg


def add_config(cfg):
    # AISFormer config
    cfg.MODEL.AISFormer = CN()
    cfg.MODEL.AISFormer.USE = True
    cfg.MODEL.AISFormer.JUSTIFY_LOSS = True

    cfg.MODEL.AISFormer.MASK_LOSS_ONLY = False

    # transformer layers 
    cfg.MODEL.AISFormer.N_LAYERS = 1
    cfg.MODEL.AISFormer.N_HEADS = 2

    # boundary loss
    cfg.MODEL.AISFormer.BO_LOSS = False

    # addation
    cfg.SOLVER.OPTIMIZER = "AdamW"

    # custom head
    cfg.MODEL.ROI_MASK_HEAD.CUSTOM_NAME = 'AISFormer'

    # custom
    cfg.DICE_LOSS = False

    # amodal eval
    cfg.MODEL.ROI_MASK_HEAD.VERSION = 0
    cfg.MODEL.AISFormer.AMODAL_EVAL = True

    # not related to aisformer, refactor and remove later
    cfg.MODEL.ALL_LAYERS_ROI_POOLING = False


if __name__== '__main__':
    cfg = get_cfg()
    add_config(cfg)
