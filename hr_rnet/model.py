import ast
import configparser

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import ops
from tensorflow.python.ops.init_ops import VarianceScaling

from .front import HRFront
from .stage import HRStage
from .transition import ExtraTransition


def he_normal_fanout(seed=None):
    """He normal initializer.
  
    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / fan_out)`
    where `fan_in` is the number of input units in the weight tensor.
    To keep aligned with official implementation
    """
    return VarianceScaling(
        scale=2., mode="fan_out", distribution="truncated_normal", seed=seed)


def load_net_cfg_from_file(cfgfile):
    def load_from_options(section, cfg):
        options = dict()
        xdict = dict(cfg.items(section))
        for key, value in xdict.items():
            try:
                value = ast.literal_eval(value)
            except:
                value = value
            options[key] = value
        return options

    cfg = configparser.ConfigParser()
    cfg.read(cfgfile)

    sections = cfg.sections()
    options = dict()
    for _section in sections:
        options[_section] = load_from_options(_section, cfg)

    return options


def HRNet(config_file, input, bn_is_training):
    cfg = load_net_cfg_from_file(config_file)
    stages = []

    front = HRFront(num_channels=cfg['FRONT']['num_channels'],
                    output_channels=[i * cfg['FRONT']['output_channels'] for i in range(1, 3)])
    stages.append(front)

    num_stages = cfg['NET']['num_stages']
    for i in range(num_stages):
        _key = 'S{}'.format(i + 1)
        if i != num_stages - 1:
            _key_next = 'S{}'.format(i + 2)
        else:
            _key_next = 'S{}'.format(i + 1)
        _stage = HRStage(stage_id=i + 1,
                         num_modules=cfg[_key]['num_modules'],
                         num_channels=cfg[_key]['num_channels'],
                         num_out_channels=cfg[_key_next]['num_channels'],
                         num_blocks=cfg[_key]['num_blocks'],
                         num_branches=cfg[_key]['num_branches'],
                         block_type=cfg[_key]['block_type'],
                         last_stage=True if i == num_stages - 1 else False)
        stages.append(_stage)
        # from stage1 to stage2, the #channel changed, so we need an extra transition layer.
        if i == 0:
            _stage = ExtraTransition(stage_id=i + 1,
                                     num_channels=cfg[_key]['num_channels'],
                                     num_branches=cfg[_key_next]['num_branches'],
                                     num_out_channels=cfg[_key_next]['num_channels'])
            stages.append(_stage)

    batch_norm_params = {'epsilon': 1e-5,
                         'scale': True,
                         'is_training': bn_is_training,
                         'updates_collections': ops.GraphKeys.UPDATE_OPS}
    with slim.arg_scope([layers.batch_norm], **batch_norm_params):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=he_normal_fanout(),
                            weights_regularizer=slim.l2_regularizer(cfg['NET']['weight_l2_scale'])):
            out = input
            for stage in stages:
                out = stage.forward(out)
                # print(len(out))
    return out
