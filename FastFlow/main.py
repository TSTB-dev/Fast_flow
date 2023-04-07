"""
訓練や評価を行うためのスクリプト．
"""

import argparse
import yaml
import constants as const
from evaluate import evaluate
from train import train
from postprocessing import postprocessing


def parse_args():
    """ArgumentParseインスタンスを作成し，コマンドライン引数を読み込む．
    Returns:
        args
    """
    parser = argparse.ArgumentParser(description="Train FastFlow_org on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, help="path to config file"
    )
    parser.add_argument('--name', type=str, help='dataset name')
    parser.add_argument("--data", type=str, help="path to dataset folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        help="category name in dataset. If 'all' was specified, train FastFlow by using all OK_Clip data",
    )
    parser.add_argument('--valid', type=str, help='validation category')
    parser.add_argument('--color', type=str, choices=['rgb', 'gray'])
    parser.add_argument('-p', '--patchsize', type=int, help='patch size. By default, patch separation will not do')
    parser.add_argument('--random', action='store_true', help='random patch sampling')
    parser.add_argument('--mask', action='store_true', help='target mask')
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument("--post", action='store_true', help='run postprocessing only')
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    parser.add_argument("--cond_dim", type=int, help="dimension of conditional information")
    parser.add_argument("--heatmap", action='store_true', help='saving heatmap on test images')
    parser.add_argument('-t', '--threshold', type=float, help='threshold')
    args = parser.parse_args()

    return args

def check_args(args):
    # 引数のチェック
    dataset_list = ['mvtec', 'jelly', 'package']
    assert args.name in dataset_list, f'利用可能なデータセットは{dataset_list}です．'

    if args.name == 'mvtec':
        assert args.category in const.MVTEC_CATEGORIES, f'MVTecにおいて利用可能なクラスは{const.MVTEC_CATEGORIES}です'
    if args.name == 'jelly':
        assert args.category in const.JELLY_CATEGORIES, f'Jellyにおいて利用可能なクラスは{const.JELLY_CATEGORIES}です'



if __name__ == "__main__":
    
    args = parse_args()
    yml_path = './parameter.yml'
    
    with open(yml_path, mode='r') as f:
        param_dict = yaml.safe_load(f)

    for key, value in param_dict.items():
        if args.__contains__(key):
            args.__setattr__(key, value)
    else:
        print(f"This parameter is somethin wrong: {key}:{value}")
    
    if args.valid is None:
        args.valid = args.category
    if args.eval:
        evaluate(args)
    elif args.post:
        postprocessing(args)
    else:
        train(args)
