import argparse
import importlib.util
import numpy as np

# from config import cfg
from core.engine import Trainer
from core.model import Model
from tfflat.utils import mem_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--cfg', type=str, dest='cfg', default='hrnet_w32_bs128_256x192_epoch210')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    return args


args = parse_args()
cfg = importlib.util.spec_from_file_location('cfg', '{}.py'.format(args.cfg))
cfg.set_args(args.gpu_ids, args.continue_train)
trainer = Trainer(Model(cfg), cfg)
trainer.train()
