"""
訓練や評価を行うためのスクリプト．
"""

import argparse
from argparse import ArgumentParser
import os

import torch
import yaml
from ignite.contrib import metrics

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
    train_dataset = dataset.MVTecDataset(
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
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
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
    ):
    """1エポック訓練する．

    Args:
        dataloader: 訓練セットのデータローダ
        model: FastFlowのインスタンス
        optimizer: Optimizerのインスタンス
        epoch: 何エポック目か
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


def eval_once(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module):
    """modelを評価する

    Args:
        dataloader: テストセットのdataloaderインスタンス
        model: modelインスタンス
    """
    # モデルを評価モードに設定
    model.eval()
    auroc_metric = metrics.ROC_AUC()

    # Anomaly_mapよりAUROCを計算
    for data, targets in dataloader:
        data, targets = data.cuda(), targets.cuda()

        with torch.no_grad():
            ret = model(data)

        outputs = ret["anomaly_map"].cpu().detach()
        outputs = outputs.flatten()
        targets = targets.flatten()
        auroc_metric.update((outputs, targets))
    auroc = auroc_metric.compute()
    print("AUROC: {}".format(auroc))


def train(args):
    """モデルを訓練する
    Args:
        args: ArgumentParserで受け取った引数
    """

    # checkpointを格納するディレクトリの作成
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    model = build_model(config)
    optimizer = build_optimizer(model)

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()

    for epoch in range(const.NUM_EPOCHS):

        # パラメータを更新
        train_one_epoch(train_dataloader, model, optimizer, epoch)

        # 一定間隔ごとにテストデータ全てを使い，性能を評価(この性能によって訓練エポックを変えると，汎化性能が楽観的な評価になる)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_once(test_dataloader, model)

        # 一定間隔ごとにモデルをsave
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )


def evaluate(args):
    """モデルの性能を評価する.
    Args:
        args: ArgumentParserが受けとったコマンドライン引数
    """
    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # モデルのビルド
    model = build_model(config)

    # モデルの状態をロード
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    # テストセットを作成
    test_dataloader = build_test_data_loader(args, config)

    # モデルを評価
    model.cuda()
    eval_once(test_dataloader, model)


def parse_args():
    """ArgumentParseインスタンスを作成し，コマンドライン引数を読み込む．
    Returns:
        args
    """
    parser = argparse.ArgumentParser(description="Train FastFlow_org on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        required=True,
        help="category name in mvtec",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
