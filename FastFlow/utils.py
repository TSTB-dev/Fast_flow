import os
import pathlib
import datetime
import json

from PIL import Image
import matplotlib.pyplot as plt
import torch

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
    pathlib.Path(save_dir).mkdir(parents=True)

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


def save_images(save_dir: str, anomaly_maps: torch.Tensor, filenames: list, image_size: int, color_mode: str, suffix: str):
    """画像を指定された形式で保存する

    Args:
        save_dir: 保存先のディレクトリ
        anomaly_maps: 各pixelに異常スコアを割り当てた入力画像と同じ形の画像 : (1, H, W)
        filenames: 画像のファイルパスのリスト
        image_size: 画像サイズ
        color_mode: 'rgb'もしくは'gray'
        suffix: 接尾辞, 'heatmap'など

    Returns:
        heatmap_dir: ヒートマップを格納するディレクトリのパス
    """

    # (B, 1, H, W) -> (B, H, W, 1)
    anomaly_maps = anomaly_maps.transpose(1, 3).cpu().numpy()

    # 元画像のファイル名にsuffixをつけて新しいファイル名を作成
    img_dir = os.path.join(save_dir, 'images')
    pathlib.Path(img_dir).mkdir(parents=True, exist_ok=True)

    for idx, filename in enumerate(filenames):
        if type(filename) == bytes:
            filename = filename.decode('utf-8')
        filename_new, ext = os.path.splitext(filename)
        filename_new = '_'.join(filename_new.split(os.path.sep)[-2:])
        filename_new = filename_new + "_" + suffix + ext
        save_path = os.path.join(img_dir, filename_new)

        # 元画像の読み込みとサイズの変換
        img_org = Image.open(filename)
        img_org = img_org.resize((image_size, image_size))

        # heatmapと元画像の合成
        alpha = 0.6
        heatmap = anomaly_maps[idx]
        plt.imshow(heatmap, cmap='jet', alpha=alpha)
        plt.imshow(img_org, alpha=1-alpha)
        plt.savefig(save_path)

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
    """

    # 各製品データを格納するディレクトリを取得
    class_dir_list = list(pathlib.Path(dataset_dir).iterdir())

    # 各製品クラスに関して繰り返す
    for class_dir in class_dir_list:

        normal_dir = class_dir / 'OK_Clip'
        anormal_dir = class_dir / 'NG_Clip'
        dir_list = [list(normal_dir.glob('*.bmp')), list(anormal_dir.glob('*.bmp'))]

        for filepath_list in dir_list:
            for path in filepath_list:
                # pathからファイル名と保存先のパスを取得
                filename = path.stem
                savename = path.parent / pathlib.Path(filename + '.jpg')

                # 画像を開き，jpeg形式で保存
                img = Image.open(path)
                img.save(savename)

                # BMP形式の画像を削除
                path.unlink(missing_ok=True)


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