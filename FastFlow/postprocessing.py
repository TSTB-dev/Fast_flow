import os
import pathlib
import datetime
import json

import yaml
import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import torch
import cv2
from ignite.contrib.metrics import ROC_AUC

import utils
import constants as const
import fastflow
import dataset


def mask_background(image_files: list, predictions: torch.Tensor) -> torch.Tensor:
    """ヒートマップから，元画像の背景にある異常箇所を除去する．
    Args:
        image_files: 画像ファイルパスのリスト
        predictions: 予測した二値化画像

    Returns:
        masked_imgs: 背景の異常を除去した予測マスク画像
    """
    threshold = 10
    mask_list = []

    for idx, path in enumerate(image_files):
        # 元画像を読み込み，二値化する
        img = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # 連結領域を算出し，背景領域のマスクを作成
        n_labels, labels, stats, centroid = cv2.connectedComponentsWithStats(bin_img)
        mask = np.where(labels == 0, 0, 1).astype(np.uint8)

        # マスクを適用する
        img_size = predictions.shape[-2:][::-1]  # cv2.resize()に渡す画像の形は(width, height)のため，反転
        mask = cv2.resize(mask, img_size)[np.newaxis, ...]

        predictions[idx] = predictions[idx] * mask
        mask_list.append(torch.from_numpy(mask))

    return predictions, torch.stack(mask_list, dim=0)


def create_prediction_map(threshold: float, heatmaps: torch.Tensor) -> torch.Tensor:
    """ヒートマップに対して後処理し，異常箇所を1, 正常箇所を0とするマスク画像を返す．
    Args:
        threshold: ピクセルごとのしきい値
        heatmaps: 予測ヒートマップ, (1, H, W)
    Returns:
        mask_image: 後処理したマスク画像
    """
    for idx, heatmap in enumerate(heatmaps):
        heatmaps[idx] = torch.where(heatmap > threshold, 1, 0)
    return heatmaps


def predict(dataloader, model, threshold: float, save_dir: str):
    """予測マスク画像を作成する．
    Args:
        dataloader: 予測に用いるデータのDataloaderインスタンス
        model: 予測に用いるモデル
        threshold: 二値化する際のしきい値
        save_dir: 保存先のディレクトリ

    Returns:

    """

    idx = 0
    model.eval()
    auroc_metric = ROC_AUC()
    image_files = dataloader.dataset.image_files
    target_list = []
    pred_list = []

    for data, targets in tqdm.tqdm(dataloader):
        data = data.cuda()
        batch_files = image_files[idx * const.BATCH_SIZE:(idx+1)*const.BATCH_SIZE]

        with torch.no_grad():
            ret = model(data)

        # anomaly_mapは正常確率に負をかけたものであるため，異常度に変換
        anomaly_map = ret['anomaly_map'].cpu().detach()
        anomaly_map = 1. + anomaly_map

        # しきい値にもとづき，二値化する
        pred_maps = create_prediction_map(threshold, anomaly_map)

        # 背景は正常とする
        pred_maps, mask = mask_background(batch_files, pred_maps)

        # 画像ごとの予測を算出(pred_maps内に一つでも0より大きい値(異常)があれば異常と判定)
        img_size = targets.shape[-2:]
        global_max_pool = torch.nn.MaxPool2d(kernel_size=img_size, stride=img_size)
        targets = global_max_pool(targets)[:, 0, 0, 0]
        preds = global_max_pool(pred_maps)[:, 0, 0, 0]
        target_list.append(targets)
        pred_list.append(preds)

        utils.save_images(save_dir, pred_maps, filenames=batch_files, image_size=pred_maps.shape[-2:], suffix='pred')
        utils.save_images(save_dir, mask, filenames=batch_files, image_size=pred_maps.shape[-2:], suffix='mask')

        idx += 1

    t = torch.stack(target_list, dim=0).flatten()
    p = torch.stack(pred_list, dim=0).flatten()
    precision, recall, f_score, support = precision_recall_fscore_support(t, p)
    cm = confusion_matrix(t, p)

    print(f"Precision: {precision}\nRecall: {recall}\nF-score: {f_score}\nConfusion Matrix: {cm}")


def postprocessing(args):

    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # 訓練のメタ情報の取得
    info_path = str(pathlib.Path(args.checkpoint).parent.parent / 'train_info.json')
    train_info = utils.get_training_info(info_path)
    save_dir = train_info['other']['save_dir']

    # 訓練時にパッチ分割されていた場合はそれに従う．
    patch_size = train_info['data']['patch_size']
    args.patchsize = patch_size

    model = fastflow.build_model(config, args)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_dataloader = dataset.build_test_data_loader(args, config)
    model.cuda()

    predict(test_dataloader, model, args.threshold, save_dir)


if __name__ == '__main__':
    heatmap = torch.rand(8, 1, 256, 256)
    pred_mask = create_prediction_map(0.5, heatmap)