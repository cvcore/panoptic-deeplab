import pathlib
import argparse
import torch

from .model import build_segmentation_model_from_cfg
from .config import config, update_config

def _update_args(model_path, cfg_path):
    parser = argparse.ArgumentParser(description='Test segmentation network with single process')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)


    args = parser.parse_args(['--cfg', cfg_path, 'TEST.MODEL_FILE', model_path])

    update_config(config, args)

def build_default_model(model_path, cfg_path):
    _update_args(model_path, cfg_path)
    model = build_segmentation_model_from_cfg(config)

    model_weights = torch.load(model_path)
    if 'state_dict' in model_weights.keys():
        model_weights = model_weights['state_dict']
        print('Evaluating a intermediate checkpoint.')
    model.load_state_dict(model_weights, strict=False)
    print('Test model loaded from {}'.format(model_path))

    return model
