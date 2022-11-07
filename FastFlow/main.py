"""
訓練や評価を行うためのスクリプト．
"""

import argparse
import datetime
import yaml
import os
import pathlib
import tqdm

import torch
import matplotlib.pyplot as plt
from ignite.contrib import metrics
from torch.utils.tensorboard import SummaryWriter

import constants as const
import dataset
import fastflow
import utils


def build_train_data_loader(args, config: dict) -> torch.utils.data.DataLoader:
    """
    Args:
        args: ArgumentParserで受け取った引数を格納するNamespaceインスタンス
        config: 事前学習済みモデルの設定ファイル(yaml形式)から読み込んだ情報を格納する辞書

    Returns:
        訓練データのDataloader
    """
    if args.name == 'mvtec':
        train_dataset = dataset.MVTecDataset(
            root=args.data,
            category=args.category,
            input_size=config["input_size"],
            is_train=True,
        )
    elif args.name == 'jelly':
        train_dataset = dataset.JellyDataset(
            root=args.data,
            category=args.category,
            input_size=config["input_size"],
            is_train=True,
        )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )


def build_test_data_loader(args, config: dict) -> torch.utils.data.DataLoader:
    """
    Args:
        args: ArgumentParserで受け取った引数を格納するNamespaceインスタンス
        config: 事前学習済みモデルの設定ファイル(yaml形式)から読み込んだ情報を格納する辞書

    Returns:
        テストデータのDataloader
    """
    if args.name == 'mvtec':
        test_dataset = dataset.MVTecDataset(
            root=args.data,
            category=args.category,
            input_size=config["input_size"],
            is_train=False,
            is_mask=args.mask,
        )
    elif args.name == 'jelly':
        test_dataset = dataset.JellyDataset(
            root=args.data,
            category=args.category,
            input_size=config["input_size"],
            is_train=False,
            is_mask=args.mask
        )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def build_model(config: dict) -> torch.nn.Module:
    """
    Args:
        config: 事前学習済みモデルの設定ファイル(yaml形式)から読み込んだ情報を格納する辞書

    Returns:
        FastFlowのインスタンス
    """

    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )

    # 学習可能なパラメータの数を取得
    print(
        "Total model Param#: {}[MB]".format(
            sum(p.numel() for p in model.parameters())/1e+6
        )
    )
    return model


def build_optimizer(model) -> torch.optim.Optimizer:
    """Optimizerをビルドする．学習率や重み減衰のパラメータはconstants.pyで指定．
    """
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )


def train_one_epoch(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> float:
    """1エポック訓練する．

    Args:
        dataloader: 訓練セットのデータローダ
        model: FastFlowのインスタンス
        optimizer: Optimizerのインスタンス
        epoch: 何エポック目か

    Returns:
        1エポックでのlossの平均
    """

    # 訓練モードに設定
    model.train()
    # 1epoch中におけるlossの平均を計算
    loss_meter = utils.AverageMeter()


    for step, data in enumerate(dataloader):
        # forward
        data = data.cuda()
        ret = model(data)
        loss = ret["loss"]

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )

    return loss_meter.avg


def eval_once(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, is_mask: bool, train_info: dict = None) -> float:
    """modelを評価する. 入力画像に対する異常のヒートマップはsave_dir内に保存．

    Args:
        dataloader: テストセットのdataloaderインスタンス
        model: modelインスタンス
        is_mask: 異常箇所のマスクがあるかどうか
        train_info: 保存先のディレクトリ
    Returns:
        auroc: AUROC
    """
    # モデルを評価モードに設定
    model.eval()
    auroc_metric = metrics.ROC_AUC()
    if train_info:
        save_dir = train_info['other']['save_dir']
        image_size = train_info['data']['image_size']

    # Anomaly_mapよりAUROCを計算
    image_files = dataloader.dataset.image_files

    idx = 0
    for data, targets in tqdm.tqdm(dataloader):
        data, targets = data.cuda(), targets.cuda()
        batch_files = image_files[idx * const.BATCH_SIZE:(idx+1)*const.BATCH_SIZE]

        with torch.no_grad():
            ret = model(data)

        outputs = ret["anomaly_map"].cpu().detach()

        # heatmapを保存
        if train_info:
            img_dir = utils.save_images(save_dir, outputs, batch_files, image_size, color_mode='rgb', suffix='heatmap')

        if is_mask:
            # pixelごとのスコアにより，AUROCを算出．
            # outputs: (B, 1, H, W) -> (B * 1 * H * W, )
            # targets: (B, 1, H, W) -> (B * 1 * H * W, )
            outputs = outputs.flatten()
            targets = targets.flatten()
            auroc_metric.update((outputs, targets))
        else:
            # imageごとのスコアにより，AUROCを算出
            # Anomaly_mapを空間方向にわたって平均
            # outputs: (B, 1, H, W) -> (B, )
            # targets: (B, )
            outputs = torch.mean(outputs, dim=[1, 2, 3])
            auroc_metric.update((outputs, targets))

        idx += 1

    auroc = auroc_metric.compute()
    print("AUROC: {}".format(auroc))
    return auroc


def train(args):
    """モデルを訓練する
    Args:
        args: ArgumentParserで受け取った引数
    """

    # logを格納するディレクトリの作成とSummaryWriterの定義
    log_dir, start_time = utils.create_log_dir("fastflow", args.category)
    print(f"TensorBoard上で学習状況を確認するには，次のコマンドを実行してください．\n tensorboard --logdir={log_dir} --port 0")
    writer = SummaryWriter(log_dir)

    # checkpointを格納するディレクトリの作成
    save_dir = utils.create_save_dir('fastflow', args.category, args.data)
    ckpt_dir = os.path.join(save_dir, 'ckpt')
    pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # backboneのメタ情報を読み込み，モデルをビルド
    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    model = build_model(config)
    optimizer = build_optimizer(model)

    # dataloaderを作成し，モデルをGPUに転送
    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()


    for epoch in range(const.NUM_EPOCHS):

        # パラメータを更新
        loss = train_one_epoch(train_dataloader, model, optimizer, epoch)
        writer.add_scalar('Loss/train', loss, epoch + 1)

        # 一定間隔ごとにテストデータ全てを使い，性能を評価(この性能によって訓練エポックを変えると，汎化性能が楽観的な評価になる)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            auroc = eval_once(test_dataloader, model, args.mask)
            writer.add_scalar('AUROC/test', auroc, epoch + 1)

        # 一定間隔ごとにモデルをsave
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(ckpt_dir, "%d.pt" % epoch),
            )
    end_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    utils.save_training_info(args, config, start_time, end_time, save_dir, log_dir, ckpt_dir)


def evaluate(args):
    """モデルの性能を評価する.
    Args:
        args: ArgumentParserが受けとったコマンドライン引数
    """
    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # 訓練のメタ情報の取得
    info_path = str(pathlib.Path(args.checkpoint).parent.parent / 'train_info.json')
    train_info = utils.get_training_info(info_path)

    # モデルのビルド
    model = build_model(config)

    # モデルの状態をロード
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    # テストセットを作成
    test_dataloader = build_test_data_loader(args, config)

    # モデルを評価
    model.cuda()
    eval_once(test_dataloader, model, args.mask, train_info=train_info)


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
