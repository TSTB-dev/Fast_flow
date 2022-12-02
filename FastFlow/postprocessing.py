import os
import pathlib
import datetime
import json

import yaml
import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, auc, roc_curve
import torch
from torchvision import transforms
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


def mask_edge(predictions: torch.Tensor, l_x: float, r_x: float) -> torch.Tensor:
    """ヒートマップから，端の部分にある異常を無視する．
    Args:
        heatmaps: heatmap
        l_x: 有効な領域の左端のx座標
        r_x: 有効な領域の右端のx座標

    Returns:
        masked_imgs: 左右端の異常を除去した予測マスク画像
    """
    outputs = torch.zeros_like(predictions)
    for idx, pred_map in enumerate(predictions):
        # mask_map: (1, H, W)
        mask_map = torch.zeros_like(pred_map)
        mask_map[..., l_x:r_x] = 1.
        outputs[idx] = mask_map * pred_map
    return outputs


def create_prediction_map(threshold: float, heatmaps: torch.Tensor) -> torch.Tensor:
    """ヒートマップに対して後処理し，異常箇所を1, 正常箇所を0とするマスク画像を返す．
    Args:
        threshold: ピクセルごとのしきい値
        heatmaps: 予測ヒートマップ, (1, H, W)
    Returns:
        mask_image: 後処理したマスク画像
    """
    outputs = torch.zeros_like(heatmaps)
    for idx, heatmap in enumerate(heatmaps):
        outputs[idx] = torch.where(heatmap > threshold, 1, 0)
    return outputs


def area_thresholding(images: torch.Tensor, area_thresh: float) -> torch.Tensor:
    """二値化された異常の候補領域(regions)のなかで，一定の面積以下のものを無視する．
    Args:
        images: 二値化された画像のバッチ, (B, 1, H, W)
        area_thresh: 面積のしきい値
    Returns:
        outputs: area_thresh以下の面積のregionを除去した二値化画像．
    """
    outputs = torch.zeros_like(images)
    for idx, image in enumerate(images):
        img_arr = image.permute([1, 2, 0]).numpy().astype(np.uint8)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_arr)
        # 背景(画素値が0)は0とラベリングされているので，異常の候補領域(region)のみ抽出, この関数についてはこちら[https://axa.biopapyrus.jp/ia/opencv/object-detection.html]
        region_stats = stats[1:]
        region_areas = region_stats[:, -1]
        valid_label_idx = np.where(region_areas > area_thresh)[0]
        if valid_label_idx.size == 0:
            pass
        else:
            for val_idx in valid_label_idx:
                outputs[idx] += (labels == val_idx).astype(np.float32)

    return outputs


def predict(dataloader, model, threshold: float, save_dir: str):
    """予測マスク画像を作成する．
    Args:
        dataloader: 予測に用いるデータのDataloaderインスタンス
        model: 予測に用いるモデル
        threshold: 二値化する際のしきい値
        save_dir: 保存先のディレクトリ

    Returns:

    """
    # しきい値を定める．
    # [ヒートマップの左右両端の無効化] l_x: 有効領域の左端のx座標, r_x: 同様に右端のx座標
    # [小さすぎる異常面積の無効化] area_thresh: 異常面積のしきい値，二値化後に適用．
    l_x, r_x = 50, 1070  # 単位はpixel
    area_thresh = 100

    idx = 0
    model.eval()
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
        # 異常マップの端の部分は無視する．
        anomaly_map = 1. + anomaly_map
        anomaly_map = mask_edge(anomaly_map, l_x, r_x)

        # しきい値が指定されている場合，しきい値にもとづき，二値化する.
        pred_maps = torch.ones_like(anomaly_map)
        if threshold:
            pred_maps = create_prediction_map(threshold, anomaly_map)
            utils.save_images(save_dir, pred_maps, filenames=batch_files, image_size=anomaly_map.shape[-2:],
                              suffix='pred')
            # pred_maps = area_thresholding(pred_maps, area_thresh)

        # 画像ごとの予測を算出（各画像のスコアはその画像のAnomaly mapの最大値とする）
        img_size = targets.shape[-2:]
        global_max_pool = torch.nn.MaxPool2d(kernel_size=img_size, stride=img_size)
        targets = global_max_pool(targets)[:, 0, 0, 0]
        preds = global_max_pool(anomaly_map * pred_maps)[:, 0, 0, 0]
        target_list.append(targets)
        pred_list.append(preds)

        # utils.save_images(save_dir, anomaly_map, filenames=batch_files, image_size=anomaly_map.shape[-2:], suffix='pred')

        idx += 1

    # TODO: p -> scoresに修正
    t = torch.stack(target_list, dim=0).flatten()
    p = torch.stack(pred_list, dim=0).flatten()

    normal_idx = torch.where(t == 0.)
    anomaly_idx = torch.where(t == 1.)
    normal_scores = p[normal_idx]
    anomaly_scores = p[anomaly_idx]

    fpr, tpr, thresholds = roc_curve(t, p)
    best_thresholds = thresholds[np.argmax(tpr-fpr)]
    auroc = auc(fpr, tpr)
    utils.save_histogram(save_dir, auroc, normal_scores, anomaly_scores)

    preds_label = torch.where(p > best_thresholds, 1., 0.)
    precision, recall, f_score, support = precision_recall_fscore_support(t, preds_label)
    cm = confusion_matrix(t, preds_label)

    print(f"Precision: {precision}\nRecall: {recall}\nF-score: {f_score}\nConfusion Matrix: {cm}")
    print(f"Image-level AUROC: {auroc}\n Best thresholds: {best_thresholds}")



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