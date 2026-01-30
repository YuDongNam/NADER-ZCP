# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Modified for NADER - Easy environment switching (Local WSL / Google Colab)
# --------------------------------------------------------

import os
import yaml
import torch
from yacs.config import CfgNode as CN

_C = CN()

# =============================================================================
# ENVIRONMENT SETTINGS (Easy switch between Local WSL and Google Colab)
# =============================================================================
# Set USE_GPU = True for Google Colab, False for local WSL/CPU
_C.USE_GPU = True
# Device will be set automatically based on USE_GPU and CUDA availability

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 32
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = './data'
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'no'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 0
# Load data to memory
_C.DATA.IMG_ON_MEMORY = False

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# # Model type
# _C.MODEL.TYPE = 'INTERN_IMAGE'
# # Model name
# _C.MODEL.NAME = 'intern_image'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# # Dropout rate
# _C.MODEL.DROP_RATE = 0.0
# # Drop path rate
# _C.MODEL.DROP_PATH_RATE = 0.1
# # Drop path type
# _C.MODEL.DROP_PATH_TYPE = 'linear'  # linear, uniform
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# # INTERN_IMAGE parameters
# _C.MODEL.INTERN_IMAGE = CN()
# _C.MODEL.INTERN_IMAGE.DEPTHS = [4, 4, 18, 4]
# _C.MODEL.INTERN_IMAGE.GROUPS = [4, 8, 16, 32]
# _C.MODEL.INTERN_IMAGE.CHANNELS = 64
# _C.MODEL.INTERN_IMAGE.LAYER_SCALE = None
# _C.MODEL.INTERN_IMAGE.OFFSET_SCALE = 1.0
# _C.MODEL.INTERN_IMAGE.MLP_RATIO = 4.0
# _C.MODEL.INTERN_IMAGE.CORE_OP = 'DCNv3'
# _C.MODEL.INTERN_IMAGE.POST_NORM = False
# _C.MODEL.INTERN_IMAGE.RES_POST_NORM = False
# _C.MODEL.INTERN_IMAGE.DW_KERNEL_SIZE = None
# _C.MODEL.INTERN_IMAGE.USE_CLIP_PROJECTOR = False
# _C.MODEL.INTERN_IMAGE.LEVEL2_POST_NORM = False
# _C.MODEL.INTERN_IMAGE.LEVEL2_POST_NORM_BLOCK_IDS = None
# _C.MODEL.INTERN_IMAGE.CENTER_FEATURE_SCALE = False
# _C.MODEL.INTERN_IMAGE.REMOVE_CENTER = False



# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20

_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# ZeRO
_C.TRAIN.OPTIMIZER.USE_ZERO = False
# freeze backbone
_C.TRAIN.OPTIMIZER.FREEZE_BACKBONE = None
# dcn lr
_C.TRAIN.OPTIMIZER.DCN_LR_MUL = None

# EMA
_C.TRAIN.EMA = CN()
_C.TRAIN.EMA.ENABLE = False
_C.TRAIN.EMA.DECAY = 0.9998

# LR_LAYER_DECAY
_C.TRAIN.LR_LAYER_DECAY = False
_C.TRAIN.LR_LAYER_DECAY_RATIO = 0.875

# FT head init weights
_C.TRAIN.RAND_INIT_FT_HEAD = False

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'
# RandomResizedCrop
_C.AUG.RANDOM_RESIZED_CROP = False
_C.AUG.MEAN = (0.485, 0.456, 0.406)
_C.AUG.STD = (0.229, 0.224, 0.225)

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 20
# eval freq
_C.EVAL_FREQ = 1
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
_C.EVAL_22K_TO_1K = False

_C.AMP_TYPE = 'float16'


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg))
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if hasattr(args, 'opts') and args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if hasattr(args, 'batch_size') and args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if hasattr(args, 'dataset') and args.dataset:
        config.DATA.DATASET = args.dataset
    if hasattr(args, 'data_path') and args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if hasattr(args, 'zip') and args.zip:
        config.DATA.ZIP_MODE = True
    if hasattr(args, 'cache_mode') and args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if hasattr(args, 'pretrained') and args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if hasattr(args, 'resume') and args.resume:
        config.MODEL.RESUME = args.resume
    if hasattr(args, 'accumulation_steps') and args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if hasattr(args, 'use_checkpoint') and args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if hasattr(args, 'amp_opt_level') and args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if hasattr(args, 'output') and args.output:
        config.OUTPUT = args.output
    if hasattr(args, 'tag') and args.tag:
        config.TAG = args.tag
    if hasattr(args, 'eval') and args.eval:
        config.EVAL_MODE = True
    if hasattr(args, 'throughput') and args.throughput:
        config.THROUGHPUT_MODE = True
    if hasattr(args, 'save_ckpt_num') and args.save_ckpt_num:
        config.SAVE_CKPT_NUM = args.save_ckpt_num
    if hasattr(args, 'use_zero') and args.use_zero:
        config.TRAIN.OPTIMIZER.USE_ZERO = True
    # set local rank for distributed training
    if hasattr(args, 'local_rank') and args.local_rank:
        config.LOCAL_RANK = args.local_rank
    if hasattr(args, 'warm_up_epochs') and args.warm_up_epochs:
        config.TRAIN.WARMUP_EPOCHS = args.warm_up_epochs
    if hasattr(args, 'epochs') and args.epochs:
        config.TRAIN.EPOCHS = args.epochs
    if hasattr(args, 'ema') and args.ema:
        config.TRAIN.EMA.ENABLE = True
    if hasattr(args, 'lr') and args.lr:
        config.TRAIN.BASE_LR = args.lr
    if hasattr(args, 'optimizer') and args.optimizer:
        config.TRAIN.OPTIMIZER.NAME = args.optimizer
    

    # output folder
    config.MODEL_NAME = args.model_name
    # config.OUTPUT = os.path.join(config.OUTPUT, args.model_name)
    config.OUTPUT = os.path.join(config.OUTPUT, config.DATA.DATASET, config.MODEL_NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


# =============================================================================
# HELPER FUNCTIONS for easy access to environment settings
# =============================================================================

def get_device():
    """
    Returns the appropriate device based on USE_GPU setting and CUDA availability.
    Usage: device = get_device()
    """
    if _C.USE_GPU and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        if _C.USE_GPU and not torch.cuda.is_available():
            print("[WARNING] GPU requested but CUDA not available. Falling back to CPU.")
        return torch.device("cpu")

def get_num_workers():
    """
    Returns the configured number of workers.
    Usage: num_workers = get_num_workers()
    """
    return _C.DATA.NUM_WORKERS

def set_environment(use_gpu=True, num_workers=4):
    """
    Convenience function to quickly set environment settings.
    
    Args:
        use_gpu: True for GPU (Colab), False for CPU (local WSL)
        num_workers: Number of data loading workers (0 for WSL, 4 for Colab)
    
    Usage:
        from train_utils.config import set_environment
        set_environment(use_gpu=True, num_workers=4)  # Colab
        set_environment(use_gpu=False, num_workers=0)  # Local WSL
    """
    _C.defrost()
    _C.USE_GPU = use_gpu
    _C.DATA.NUM_WORKERS = num_workers
    _C.freeze()
    print(f"[Config] USE_GPU={use_gpu}, NUM_WORKERS={num_workers}, DEVICE={get_device()}")

def print_environment():
    """Print current environment configuration."""
    print("=" * 50)
    print("NADER Environment Configuration")
    print("=" * 50)
    print(f"  USE_GPU: {_C.USE_GPU}")
    print(f"  DEVICE: {get_device()}")
    print(f"  NUM_WORKERS: {_C.DATA.NUM_WORKERS}")
    print("=" * 50)
