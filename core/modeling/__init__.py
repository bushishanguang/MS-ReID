# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline


def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    model = Baseline(
        num_classes,
        cfg.MODEL.LAST_STRIDE,
        cfg.MODEL.PRETRAIN_PATH,
        cfg.MODEL.NECK,
        cfg.TEST.NECK_FEAT,
        cfg.MODEL.NAME,
        cfg.MODEL.PRETRAIN_CHOICE,
        use_wavelet=cfg.MODEL.USE_WAVELET,
        use_multiscale_branch=cfg.MODEL.USE_MULTISCALE_BRANCH,
        use_attention_fusion=cfg.MODEL.USE_ATTENTION_FUSION,
        return_visuals=cfg.MODEL.RETURN_VISUALS,
    )
    return model
