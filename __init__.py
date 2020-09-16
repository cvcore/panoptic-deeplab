import pathlib
import argparse
import torch

from segmentation.model import build_segmentation_model_from_cfg
from segmentation.config import config, update_config

CFG_PATH_DEFAULT = str(pathlib.Path(__file__).parent.joinpath('configs').joinpath('panoptic_deeplab_R50_os32_cityscapes.yaml').resolve())
MODEL_PATH_DEFAULT = str(pathlib.Path(__file__).parent.joinpath('saved_models').joinpath('panoptic_deeplab_R50_os32_cityscapes.pth').resolve())

def _use_default_args():
    parser = argparse.ArgumentParser(description='Test segmentation network with single process')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)


    args = parser.parse_args(['--cfg', CFG_PATH_DEFAULT, 'TEST.MODEL_FILE', MODEL_PATH_DEFAULT])

    update_config(config, args)

def build_default_model():
    _use_default_args()
    model = build_segmentation_model_from_cfg(config)

    model_weights = torch.load(MODEL_PATH_DEFAULT)
    if 'state_dict' in model_weights.keys():
        model_weights = model_weights['state_dict']
        print('Evaluating a intermediate checkpoint.')
    model.load_state_dict(model_weights, strict=True)
    print('Test model loaded from {}'.format(MODEL_PATH_DEFAULT))

    return model