import os
import pathlib
import datetime
import json

import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

import constants as const



class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_save_dir(model_name: str, class_name: str, dataset_path: str) -> str:
    """訓練のメタ情報やモデルを格納するためのディレクトリを作成する．

    Args:
        model_name: モデルの名称
        class_name: データのクラス名 (たとえば，'pill')
        dataset_path: データセットのパス(MVTecの場合，'*/data/mvtec'などデータセットディレクトリのパス)

    Returns:
        saveディレクトリのパス, 訓練の終了時刻

    """
    dataset_name = dataset_path.split(os.path.sep)[-1]
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = os.path.join(
        os.getcwd(),
        "results",
        dataset_name,
        class_name,
        model_name,
        f"{now}"
    )
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    return save_dir


def create_log_dir(model_name: str, class_name: str) -> (str, str):
    """Tensorboardのためログディレクトリを作成する．

    Args:
        model_name: モデルの名称
        class_name: データのクラス名(たとえば，'hazelnut')

    Returns:
        ログディレクトリのパス, 訓練の開始時刻

    """
    # 現在時刻を取得
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    dirname = now + '_' + class_name + '_' + model_name

    log_dir = os.path.join(
        os.getcwd(),
        "logs",
        dirname
    )
    pathlib.Path(log_dir).mkdir(parents=True)

    return log_dir, now


def save_images(save_dir: str, images: torch.Tensor, filenames: list, image_size: int, patch_size: int = None, color_mode: str = 'rgb', suffix: str = 'heatmap', class_name: str = ''):
    """画像を指定された形式で保存する

    Args:
        save_dir: 保存先のディレクトリ
        images: 保存する画像
        filenames: 画像のファイルパスのリスト
        image_size: 画像サイズ
        patch_size: パッチサイズ(訓練時にパッチ分割されていた場合)
        color_mode: 'rgb'もしくは'gray'
        suffix: 接尾辞, 'heatmap'など
        class_name: 評価に用いるクラス名

    Returns:
        heatmap_dir: ヒートマップを格納するディレクトリのパス
    """


    if patch_size:
        # (B, N, 1, P, P) -> (B, N, P, P, 1)
        images = images.permute([0, 1, 3, 4, 2]).cpu().numpy()
    else:
        # (B, 1, H, W) -> (B, H, W, 1)
        images = images.permute([0, 2, 3, 1]).cpu().numpy()

    # 元画像のファイル名にsuffixをつけて新しいファイル名を作成
    img_dir = os.path.join(save_dir, f'{suffix}_{class_name}')
    pathlib.Path(img_dir).mkdir(parents=True, exist_ok=True)

    for idx, filename in enumerate(filenames):
        if type(filename) == bytes:
            filename = filename.decode('utf-8')
        filename_new, ext = os.path.splitext(filename)
        filename_new = '_'.join(filename_new.split(os.path.sep)[-2:])
        filename_new = filename_new + ext
        save_path = os.path.join(img_dir, filename_new)

        # 元画像の読み込みとサイズの変換
        img_org = Image.open(filename)
        img_org = img_org.resize(size=image_size[::-1])  # torchは(H, W)だがPillowは(W, H)であるため．image_sizeは(H, W)

        # パッチに分割されている場合は，heatmapを結合する．
        if patch_size:
            heatmap = convert_patch_to_image(images[idx], image_size, patch_size)
        else:
            heatmap = images[idx]

        # heatmapと元画像の合成
        if suffix == 'pred':
            alpha = 1.0
            cmap = 'gray'
        else:
            alpha = 0.6
            cmap = 'jet'

        fig = plt.figure()
        plt.imshow(heatmap, cmap=cmap, alpha=alpha)
        plt.imshow(img_org, alpha=1-alpha)
        plt.savefig(save_path)
        plt.close(fig)

    return img_dir


def save_training_info(args, config: dict, start_time: str, end_time: str, save_dir: str, log_dir: str, ckpt_dir: str) -> str:
    """訓練のメタ情報をjsonファイルに保存します．

    Args:
        args: ArgParserで受け取ったコマンドライン引数を格納するオブジェクト
        config: Backboneネットワークのメタ情報を格納する辞書
        start_time: 訓練の開始時刻
        end_time: 訓練の終了時刻
        save_dir: 訓練情報を保存するディレクトリ
        log_dir: logを格納するディレクトリ
        ckpt_dir: modelのcheckpointを格納するディレクトリ

    Returns:
        info_path: jsonファイルのパス
    """

    info_dict = {
        'data': {
            'dataset_path': args.data,
            'data_class': args.category,
            'color_mode': args.color,
            'image_size': config['input_size'],
            'patch_size': args.patchsize
        },
        'hyperparameters': {
            'n_epochs': const.NUM_EPOCHS,
            'batch_size': const.BATCH_SIZE,
            'learning_rate': const.LR,
            'weight_decay': const.WEIGHT_DECAY,
        },
        'result': {
            'start_time': start_time,
            'end_time': end_time,
        },
        'other': {
            'log_dir': log_dir,
            'save_dir': save_dir,
            'ckpt_dir': ckpt_dir,
            'log_interval': const.LOG_INTERVAL,
            'eval_interval': const.EVAL_INTERVAL,
            'ckpt_interval': const.CHECKPOINT_INTERVAL
        }
    }

    # jsonファイルとして書き出し
    info_path = os.path.join(save_dir, f'train_info.json')
    pathlib.Path(info_path).touch()
    with open(info_path, 'w') as f:
        json.dump(info_dict, f, indent=4)

    return info_path


def save_evaluate_info(args, save_dir: str, auc: float, threshold: float, fpr: float, tpr: float):
    """評価時のメタ情報を保存
    Args:
        args: ArgParseが受けとった引数
        save_dir: 保存先のディレクトリ
        auc: AUCスコア
        threshold: 最良のしきい値
        fpr: 上のしきい値におけるFPR
        tpr: 上のしきい値におけるTPR

    Returns:
    """
    eval_metrics = 'Pixel-level AUC' if args.mask else 'Image-level AUC'

    info_dict = {
        'evaluation_metrics': eval_metrics,
        'AUC score': float(auc),
        'Best threshold': float(threshold),
        'False Positive Rate': float(fpr),
        'True Positive Rate': float(tpr),
    }

    info_path = pathlib.Path(os.path.join(save_dir, 'eval_info_399.json'))
    info_path.touch()
    with open(info_path, mode='w') as f:
        json.dump(info_dict, f, indent=4)

    return info_path


def save_histogram(save_dir: str, auroc: float, normal_scores: list, anomaly_scores: list):
    """正常・異常サンプルのスコアを横軸にとるヒストグラムを作成し，保存．
    Args:
        save_dir: 保存先のディレクトリ
        auroc: AUROCスコア
        normal_scores: 正常サンプルのスコア
        anomaly_scores: 異常サンプルのスコア

    Returns:
    """

    fig_path = os.path.join(save_dir, 'histogram.png')
    figure = plt.figure()
    plt.title(f'Histogram (AUROC: {auroc})')
    plt.xlabel('Anomaly score')
    plt.ylabel('Number of samples')
    plt.hist([normal_scores, anomaly_scores], bins=50)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(figure)


def convert_bmp_to_jpg(dataset_dir: str):
    """
    データセット内のBMP形式の画像をJPEG形式に変換する．
    Args:
        dataset_dir: データセットディレクトリのパス．直下には各製品データを格納するディレクトリが必要．

        dataset_dir/
            ├── classA
            │   ├── NG_Clip
                     ├── NG_1.BMP
                     │・・・
                     │── NG_99.BMP
            │   └── OK_Clip
            └── classB
                ├── NG_Clip
                └── OK_Clip
        NG_Clip_Labelがクラスディレクトリ直下にある場合は，ラベルも変換．

    """

    # 各製品データを格納するディレクトリを取得
    class_dir_list = list(pathlib.Path(dataset_dir).iterdir())

    # 各製品クラスに関して繰り返す
    for class_dir in tqdm.tqdm(class_dir_list):

        normal_dir = class_dir / 'OK_Clip'
        anormal_dir = class_dir / 'NG_Clip'
        label_dir = class_dir / 'NG_Clip_Label'

        dir_list = [list(normal_dir.glob('*.bmp')), list(anormal_dir.glob('*.bmp'))]
        if label_dir.exists():
            dir_list.append(list(label_dir.glob('*.bmp')))

        for filepath_list in dir_list:
            for path in filepath_list:
                # pathからファイル名と保存先のパスを取得
                filename = path.stem
                savename = path.parent / pathlib.Path(filename + '.jpg')

                # 画像を開き，jpeg形式で保存
                with Image.open(path) as img:
                    img.save(savename)

                # BMP形式の画像を削除


def get_training_info(info_path: str) -> dict:
    """訓練のメタ情報を取得．

    Args:
        info_path: メタ情報を格納するファイルのパス

    Returns:
        dict: 訓練のメタ情報を格納する辞書
    """
    with open(info_path, "r") as f:
        info = json.load(f)
    return info


def convert_patch_to_image(patches: np.ndarray, img_size: int, patch_size: int) -> torch.Tensor:
    """パッチを結合し，元の画像に戻します．
    Args:
        patches: パッチ, (N, P, P, 1)
        img_size: 元画像のサイズ
        patch_size: パッチサイズ
    Returns:
        image: パッチを結合した元の画像
    """

    # 入力のチェック
    assert img_size % patch_size == 0, "元画像のサイズはパッチサイズで割りきれるようにしてください．"
    patches = torch.from_numpy(np.squeeze(patches))  # Ndarray(N, P, P, 1) -> torch.Tensor(N, P, P)

    # 水平方向のパッチ数
    num_patch_row = img_size // patch_size
    col_patch_list = []

    for i in range(num_patch_row):
        row_patch_list = []
        for j in range(num_patch_row):
            row_patch_list.append(patches[i * num_patch_row + j])  # -> (num_patch_row], P, P)
        img_row = torch.concat(row_patch_list, dim=1)  # -> (P, image_size)
        col_patch_list.append(img_row)   # -> (num_patch_row], P, image_size)
    image = torch.concat(col_patch_list, dim=0)  # -> (image_size, image_size)
    return image


def convert_binary_image(dataset_dir: str):
    """二値の画像を[0, 255]に変換する．
    Args:
        dataset_dir: データセットディレクトリのパス
    """
    dataset_dir = pathlib.Path(dataset_dir)
    class_dirs = list(dataset_dir.iterdir())

    for class_dir in class_dirs:
        data_dir = class_dir / 'NG_Clip_Label'
        img_files = list(data_dir.glob('*.jpg'))
        for img_file in tqdm.tqdm(img_files):
            img_file = str(img_file)
            with Image.open(img_file) as img:
                img = np.asarray(img, dtype=np.uint8)
                img = np.where(img > 0, 255, 0)
                cv2.imwrite(img_file, img)


if __name__ == '__main__':
    dataset_dir = "C:\\Users\had-int22\PycharmProjects\Pytorch_AD\data\jelly_mask"
    # convert_binary_image(dataset_dir)
    convert_bmp_to_jpg(dataset_dir)
