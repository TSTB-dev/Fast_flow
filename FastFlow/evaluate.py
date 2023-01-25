import yaml
import pathlib
import tqdm
import time
from ast import literal_eval

import torch
from ignite.contrib.metrics import ROC_AUC
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import numpy as np

import constants as const
import dataset
import fastflow
import utils


def eval_once(args, dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, is_mask: bool, train_info: dict = None, save_img: bool = False) -> float:
    """modelを評価する. 入力画像に対する異常のヒートマップはsave_dir内に保存．

    Args:
        args: ArgParserが受けとった引数
        dataloader: テストセットのdataloaderインスタンス
        model: modelインスタンス
        is_mask: 異常箇所のマスクがあるかどうか
        train_info: 保存先のディレクトリ
        save_img: heatmapを保存するかどうか
    Returns:
        auroc: AUROC
    """

    # モデルを評価モードに設定
    model.eval()
    auroc_metric = ROC_AUC()
    save_dir = ''
    image_size = None

    if train_info:
        save_dir = train_info['other']['save_dir']
        image_size = literal_eval(train_info['data']['image_size'])
    image_files = dataloader.dataset.image_files

    idx = 0
    targets_list = []
    preds_list = []
    elapsed_time_list = []
    for data, targets in tqdm.tqdm(dataloader):
        data, targets = data.cuda(), targets.cuda()
        batch_files = image_files[idx * const.BATCH_SIZE:(idx+1)*const.BATCH_SIZE]

        with torch.no_grad():
            start_time = time.perf_counter()
            ret = model(data)
            end_time = time.perf_counter()
            elapsed_time_list.append((end_time - start_time)/const.BATCH_SIZE)

        if args.method == 'differnet':
            outputs = ret['probs']
            auroc_metric.update((outputs, targets))
        elif args.method == 'fastflow':
            # outputsは各pixelについて予測した正常である確率に負を掛けたもの
            outputs = ret["anomaly_map"].cpu().detach()
            auroc_metric, outputs = postprocces(args, train_info, save_img, save_dir, outputs, batch_files, image_size, model, is_mask, targets, auroc_metric)

        targets_list.append(targets)
        preds_list.append(outputs)
        idx += 1

    # 推論速度を評価する
    # 最初のバッチに対する推論はGPUへのロードを含み遅くなるので，省いている．
    mean = np.mean(elapsed_time_list[1:]) * 1000
    std = np.std(elapsed_time_list[1:]) * 1000
    print(f"Mean inference time per image: {mean:.2f}[ms]")
    print(f"Std inference time per image: {std:.2f}[ms]")

    # 予測性能を評価する．
    # 全体の予測結果を集約する．predsは正常確率に負を掛けたもののリストであるため，異常度(scores)に変換する．
    preds = torch.concat(preds_list, dim=0).cpu().numpy()
    targets = torch.concat(targets_list, dim=0).cpu().numpy()
    scores = 1. + preds

    # TPR, FPR, しきい値の計算
    fpr_list, tpr_list, thresholds = metrics.roc_curve(targets, scores)
    best_thresh_idx = np.argmax(tpr_list - fpr_list)  # TPR - FPRが最大になるようなしきい値が最良
    best_threshold = thresholds[best_thresh_idx]

    fpr, tpr = fpr_list[best_thresh_idx], tpr_list[best_thresh_idx]
    preds_sparse = np.where(scores > best_threshold, 1, 0)  # そのしきい値を用いて異常・正常を分類

    # 異常・正常サンプルのスコアを横軸にとるヒストグラムを作成
    normal_idx = np.where(targets == 0)
    anomaly_idx = np.where(targets == 1)
    normal_scores = scores[normal_idx]
    anomaly_scores = scores[anomaly_idx]

    # 混同行列，AUCの計算
    cm = metrics.confusion_matrix(targets, preds_sparse)

    auroc = auroc_metric.compute()
    print(f"AUROC: {auroc:.3f}")
    if train_info:
        utils.save_evaluate_info(args, save_dir, auroc, best_threshold, fpr, tpr)
        utils.save_histogram(save_dir, auroc, normal_scores, anomaly_scores)

    print(f"ConfusionMatrix: \n{cm}")
    print(f"best_threshold: \n{best_threshold}")
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
    # debug_on_train_data(train_dataloader, model, train_info)
    eval_once(args, test_dataloader, model, is_mask=args.mask, train_info=train_info, save_img=args.heatmap)


def postprocces(args, train_info, save_img, save_dir, outputs, batch_files, image_size, model, is_mask, targets,
                auroc_metric):
    # heatmapを保存
    save = True
    if train_info and save_img:
        if model.patch_size:
            utils.save_images(save_dir, outputs, batch_files, image_size, patch_size=model.patch_size, color_mode='rgb',
                              suffix='heatmap', class_name=args.valid)
        else:
            for path in batch_files:
                dir_name = path.parent.name  # 正常なら'OK_Clip', 異常なら'NG_Clip'
                if 'NG' in dir_name:
                    save = True
            if save:
                utils.save_images(save_dir, outputs, batch_files, image_size, color_mode='rgb', suffix='heatmap',
                                  class_name=args.valid)

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
            # max_patch_mean = torch.max(torch.mean(outputs, dim=[2, 3, 4]), dim=1)
            # outputs = max_patch_mean.values
            outputs = torch.mean(outputs, dim=[1, 2, 3, 4])
        else:
            # outputs: (B, 1, H, W) -> (B, )
            # targets: (B, )
            outputs = torch.mean(outputs, dim=[1, 2, 3])
        auroc_metric.update((outputs, targets))

    return auroc_metric, outputs


def debug_on_train_data(train_dataloder, model, train_info):
    idx = 0
    save_dir = './debug'
    image_size = literal_eval(train_info['data']['image_size'])
    image_files = train_dataloder.dataset.image_files
    model.eval()

    for data in tqdm.tqdm(train_dataloder):
        batch_files = image_files[idx * const.BATCH_SIZE:(idx+1)*const.BATCH_SIZE]
        data = data.cuda()
        with torch.no_grad():
            ret = model(data)
        outputs = ret['anomaly_map'].cpu().detach()
        utils.save_images(save_dir, outputs, batch_files, image_size, color_mode='rgb', suffix='heatmap')

        idx += 1

