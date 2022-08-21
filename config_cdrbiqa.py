

import os
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 16
_C.DATA.IMG_SIZE = 224
_C.DATA.TRAIN_PATCH_NUMBER = 1
_C.DATA.TEST_PATCH_NUMBER = 10
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 1

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'CDR-BIQA'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 6
_C.TRAIN.WARMUP_EPOCHS = 1
_C.TRAIN.WEIGHT_DECAY = 5e-4
_C.TRAIN.BASE_LR = 2e-5
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 1e-7

# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# could be overwritten by command line argument

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = 'O1'
# Path to output folder, overwritten by command line argument
_C.OUTPUT = 'output'
# Tag of experiment, overwritten by command line argument
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.TRAIN_PRINT_FREQ = 2000
_C.TEST_PRINT_FREQ = 200
# Fixed random seed
_C.SEED = 0

def update_config(config):
    config.defrost()
    # merge from specific argument
    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME)
    config.freeze()

def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config)

    return config
