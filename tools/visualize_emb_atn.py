import torch.nn as nn
import os
from detetron2.modeling.roi_heads.aistr.aistr_enfeat import AISTREncodeFeat
from tools.add_config import add_config
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec

def main():
    model_path = ""
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(args.config_file)
    aistr_model = AISTREncodeFeat(cfg, ShapeSpec(
            channels=256, width=14, height=14
        ))

    pass

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    from detectron2.config import CfgNode as CN
    cfg.DICE_LOSS = CN()
    cfg.DICE_LOSS = False
    add_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
