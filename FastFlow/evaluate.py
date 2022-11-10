import yaml
import pathlib
import tqdm

import torch
from ignite.contrib.metrics import ROC_AUC
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import numpy as np

import constants as const
import dataset
import fastflow
import utils


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
    auroc_metric = ROC_AUC()
    if train_info:
        save_dir = train_info['other']['save_dir']
        image_size = train_info['data']['image_size']

    # Anomaly_mapよりAUROCを計算
    image_files = dataloader.dataset.image_files

    idx = 0
    targets_list = []
    preds_list = []
    for data, targets in tqdm.tqdm(dataloader):
        data, targets = data.cuda(), targets.cuda()
        batch_files = image_files[idx * const.BATCH_SIZE:(idx+1)*const.BATCH_SIZE]

        with torch.no_grad():
            ret = model(data)

        # outputsは各pixelについて予測した正常である確率に負を掛けたもの
        outputs = ret["anomaly_map"].cpu().detach()

        # heatmapを保存
        if train_info:
            if model.patch_size:
                utils.save_images(save_dir, outputs, batch_files, image_size, patch_size=model.patch_size, color_mode='rgb', suffix='heatmap')
            else:
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

            if model.patch_size:
                # outputs: (B, N, 1, P, P) -> (B, )
                # targets: (B, )
                # パッチごとに異常スコアの平均をとり，それが最大のパッチの異常スコアを最終的なその画像のスコアとして扱う．
                max_patch_mean = torch.max(torch.mean(outputs, dim=[2, 3, 4]), dim=1)
                outputs = max_patch_mean.values

            else:
                # outputs: (B, 1, H, W) -> (B, )
                # targets: (B, )
                outputs = torch.mean(outputs, dim=[1, 2, 3])

            targets_list.append(targets)
            preds_list.append(outputs)
            auroc_metric.update((outputs, targets))

        idx += 1

    # 結果を評価する．
    # 全体の予測結果を集約する．predsは正常確率に負を掛けたもののリストであるため，異常確率(scores)に変換する．
    preds = torch.concat(preds_list, dim=0).cpu().numpy()
    targets = torch.concat(targets_list, dim=0).cpu().numpy()
    scores = 1. + preds

    # TPR, FPR, しきい値の計算
    fpr, tpr, thresholds = metrics.roc_curve(targets, scores)
    best_thresh_idx = np.argmax(tpr - fpr)  # TPR - FPRが最大になるようなしきい値が最良
    best_threshold = thresholds[best_thresh_idx]
    preds_sparse = np.where(scores > best_threshold, 1, 0)  # そのしきい値を用いて異常正常を分類

    # 混同行列，AUCの計算
    cm = metrics.confusion_matrix(targets, preds_sparse)
    auc = metrics.auc(fpr, tpr)

    auroc = auroc_metric.compute()
    print("AUROC: {}".format(auroc))

    print(f"AUROC(confirm): {auc}")
    print(f"ConfusionMatrix: \n{cm}")
    return auroc


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

    # 訓練時にパッチ分割されていた場合はそれに従う．
    patch_size = train_info['data']['patch_size']
    args.patchsize = patch_size

    # モデルのビルド
    model = fastflow.build_model(config, args)

    # モデルの状態をロード
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    # テストセットを作成
    test_dataloader = dataset.build_test_data_loader(args, config)

    # モデルを評価
    model.cuda()
    eval_once(test_dataloader, model, args.mask, train_info=train_info)

