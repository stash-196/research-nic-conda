import os
from yacs.config import CfgNode as CN

_C = CN()
_C.SEED = 0

_C.SYSTEM = CN()

_C.TRAIN = CN()
_C.TRAIN.LR = 0.1


def get_config(cfg_file=None):
    _C.defrost()
    if cfg_file and os.path.exists(cfg_file):
        _C.merge_from_file(cfg_file)
    else:
        print("cfg_file %s not found, using default settings." % cfg_file)
    _C.freeze()
    return _C.clone()


def update_config(cfg_file=None, cfg_object=None):
    _C.defrost()
    if cfg_file and os.path.exists(cfg_file):
        _C.merge_from_file(cfg_file)
    else:
        print("cfg_file %s not found, using default settings." % cfg_file)
    _C.freeze()
    return _C.clone()
