import os

from parsers import get_parser
from model.model_utils import CKPT_PATH, setseed, ModelSave

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def averaging(args):
    ckpt_dir = os.path.join(CKPT_PATH, args.save_name)
    ckpt_dir = os.path.join(ckpt_dir, args.ckpt_path)
    model_save = ModelSave(
        ckpt_dir=ckpt_dir,
        const_save=[],
        w1=0.9
    )
    
    #----------------------------------start averaging----------------------------
    model_save.model_average(
        step_list=args.average_list,
        model_name=args.average_name
    )


if __name__ == '__main__':
    parser = get_parser(mode='average')
    args = parser.parse_args()
    #----------------enter your ckpt path and ckpt name----------------#
    args.ckpt_path = ""
    args.average_list = []
    setseed(args.seed)
    averaging(args)