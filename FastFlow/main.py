"""
訓練や評価を行うためのスクリプト．
"""

import argparse

import constants as const
from evaluate import evaluate
from train import train


def parse_args():
    """ArgumentParseインスタンスを作成し，コマンドライン引数を読み込む．
    Returns:
        args
    """
    parser = argparse.ArgumentParser(description="Train FastFlow_org on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument('--name', type=str, required=True, help='dataset name')
    parser.add_argument("--data", type=str, required=True, help="path to dataset folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        required=True,
        help="category name in mvtec",
    )
    parser.add_argument('--color', type=str, choices=['rgb', 'gray'])
    parser.add_argument('-p', '--patchsize', type=int, help='patch size. By default, patch separation will not do')
    parser.add_argument('--mask', action='store_true', help='whether target mask is exists')
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    args = parser.parse_args()

    # 引数のチェック
    dataset_list = ['mvtec', 'jelly']
    assert args.name in dataset_list, f'利用可能なデータセットは{dataset_list}です．'

    if args.name == 'mvtec':
        assert args.category in const.MVTEC_CATEGORIES, f'MVTecにおいて利用可能なクラスは{const.MVTEC_CATEGORIES}です'
    if args.name == 'jelly':
        assert args.category in const.JELLY_CATEGORIES, f'Jellyにおいて利用可能なクラスは{const.JELLY_CATEGORIES}です'

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
