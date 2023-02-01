"""
datasetを準備するモジュール．
Pytorchのtorch.utils.data APIを使用．詳細は公式チュートリアル[https://pytorch.org/tutorials/beginner/basics/data_tutorial.html]を参照．
"""

import os
from glob import glob
from ast import literal_eval
import pathlib
import random

import numpy as np
from numpy.random import default_rng
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode

import constants as const


class MVTecDataset(torch.utils.data.Dataset):
    """
    MVTecDatasetをロードするためのクラス．
    """

    def __init__(self, root: str, category: str, input_size: int, is_train: bool = True, is_mask: bool = False, color: str = 'rgb'):
        """
        Args:
            root: MVTecADのルートディレクトリ．このディレクトリ直下に各クラスのデータディレクトリを含む．
            category: クラス名. Ex. 'hazelnut'
            input_size: 事前学習済みモデルの入力画像の形．256と指定された場合は(256, 256)の画像を表す．
            is_train: 訓練データかどうか
            is_mask: 異常箇所のマスクがあるかどうか．
            color: rgbもしくはgray
        """

        # 事前学習済みモデルに入力するための画像変換を定義
        # ImageNetによる事前学習を想定しているため，RGBの3チャンネルを持つときは，ImageNet用の正規化をする．詳細は[https://teratail.com/questions/295871]を参照
        transforms_list = [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        self.image_transform = transforms.Compose(
            transforms_list
        )

        # 訓練データのパスのリスト取得
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        # テストデータのパスのリスト取得とtarget(異常箇所の正解マスク画像)に対する変換を取得
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            if is_mask:
                self.target_transform = transforms.Compose(
                    [
                        transforms.Resize(input_size),
                        transforms.ToTensor(),
                    ]
                )
            else:
                self.label = []
                for image_file in self.image_files:
                    status = image_file.split(os.path.sep)[-2]
                    if status == 'good':
                        self.label.append(0)
                    else:
                        self.label.append(1)

        self.color = color
        self.is_train = is_train
        self.is_mask = is_mask

    def __getitem__(self, index):
        """
        データセットからindexで指定されたデータを取得する．dataset[idx]のようにリストから値を取得するように扱える．
        Args:
            index: データのindex

        Returns:
            output:
                if self.is_train: 事前学習済みモデルへの入力画像
                else: 事前学習済み画像への入力画像とtarget画像(異常箇所の正解マスク画像)
        """

        # 指定されたindexの画像を読み込み，事前学習済みモデルに対応する形式に変換
        # 変換後のimageの形は，(C, input_size, input_size). PytorchはChannel-firstフォーマットをとる．(TensorFlowはChannel-lastフォーマット)
        image_file = self.image_files[index]
        image = Image.open(image_file)

        if self.color == 'gray':
            image = image.convert('RGB')
        image = self.image_transform(image)

        if self.is_train:
            return image

        else:
            if not self.is_mask:
                return image, self.label[index]
            # 正常画像に対する正解マスク画像の作成, targetは二値画像なので形は(1, input_size, input_size)
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])

            # 異常画像に対する正解マスク画像の作成と変換
            else:
                sep = os.path.sep
                target = Image.open(
                    image_file.replace(f"{sep}test{sep}", f"{sep}ground_truth{sep}").replace(
                        ".png", "_mask.png"
                    )
                )
                target = self.target_transform(target)
            return image, target

    def __len__(self):
        """
        データセットの長さを返す．len(dataset)のようにして取得可能．
        Returns: データセット内のデータ数
        """
        return len(self.image_files)


class JellyDataset(torch.utils.data.Dataset):
    """
    JellyDatasetをロードするためのクラス
    """

    def __init__(self, root: str, category: str, valid_category: str, input_size: int, is_train: bool = True, test_ratio: float = 0.1,
                 is_mask: bool = False, patch_size: int = None, random_sampling: bool = False, seed: int = 42):
        """
        Args:
            root: JellyDatasetのルートディレクトリ．このディレクトリ直下に各クラスのデータディレクトリを含む．
            category: クラス名. Ex. 'hair1'
            input_size: 事前学習済みモデルの入力画像の形．256と指定された場合は(256, 256)の画像を表す．
            is_train: 訓練データかどうか
            test_ratio: 全体のデータのうち，何割を評価用に使うか
            is_mask: 異常箇所のマスクがあるかどうか.
            patch_size: 画像をパッチに分割する際のパッチサイズ. input_size % patch_size = 0．パッチは重ならないように分割される．
            random_sampling: パッチ分割した画像をランダムに学習に用いる．ここで，パッチは重なっても良い．
            seed: 正常画像から訓練に使うものと評価に使うものを分割する際のランダムシード

        Attributes:
            self.image_transform: 画像に対するpreprocessing
            self.image_files: 画像のファイルパスのリスト
            self.label: 評価データにおける異常(1),正常(0)のラベル
            self.target_transform: 正解マスク画像に対するpreprocessing
        """
        self.input_size = input_size
        self.patch_size = patch_size

        # 事前学習済みモデルに入力するための画像変換を定義
        # ImageNetによる事前学習を想定しているため，ImageNet用の正規化をする．詳細は[https://teratail.com/questions/295871]を参照
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # 正常・異常データのパスのリスト取得
        random.seed(seed)
        if category == 'all':
           data_dir = pathlib.Path(root)
           normal_list = list(data_dir.glob('*/OK_Clip/*.jpg'))
        else:
            normal_dir = pathlib.Path(os.path.join(root, category, 'OK_Clip'))
            normal_list = list(normal_dir.glob('*.jpg'))

        anomal_dir = pathlib.Path(os.path.join(root, valid_category, 'NG_Clip'))
        anomal_list = list(anomal_dir.glob('*.jpg'))

        # 正常画像をシャッフルし，test_ratioで指定された数の正常画像以外を訓練データとして利用
        random.shuffle(normal_list)

        if is_train:
            self.image_files = normal_list[int(len(normal_list) * test_ratio):]
        else:
            # 正常画像の一部を評価用のデータとして利用
            self.image_files = normal_list[:int(len(normal_list) * test_ratio)]
            # ラベルを作成
            self.label = [0] * len(self.image_files)

            # 異常画像を評価用のデータとして利用
            self.image_files += anomal_list
            self.label += [1] * len(anomal_list)

            # 異常マスクがある場合には，マスクに対する変換も定義
            if is_mask:
                self.target_transform = transforms.Compose(
                    [
                        transforms.Resize(input_size),
                    ]
                )

        self.is_mask = is_mask
        self.is_train = is_train
        self.random = random_sampling

    def __getitem__(self, index):
        """
        データセットからindexで指定されたデータを取得する．dataset[idx]のようにリストから値を取得するように扱える．
        Args:
            index: データのindex

        Returns:
            output:
                if self.is_train: 事前学習済みモデルへの入力画像
                else: 事前学習済み画像への入力画像とtarget画像(異常箇所の正解マスク画像), マスクがない場合はラベル(0, 1)
        """
        # 指定されたindexの画像を読み込み，事前学習済みモデルに対応する形式に変換
        # -> (C, input_size, input_size)
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)

        if self.patch_size and not self.is_train:
            image = patch_split(image, self.patch_size)
        if self.patch_size and self.is_train:
            image = patch_split(image, self.patch_size, random_sampling=self.random)

        # 訓練時
        if self.is_train:
            return image

        # 評価時[マスクがない場合]
        if not self.is_mask:
            return image, self.label[index]

        # 評価時[マスクがある場合]
        # 正常画像に対する正解マスク画像の作成, targetは二値画像なので形は(1, input_size, input_size)
        if os.path.dirname(image_file).endswith("OK_Clip"):
            target = torch.zeros([1, *self.input_size])

        # 異常画像に対する正解マスク画像の作成と変換
        else:
            sep = os.path.sep
            target = Image.open(
                str(image_file).replace("NG_Clip", "NG_Clip_Label")
            )
            target = np.asarray(target, dtype=np.uint8) / 255
            target = torch.from_numpy(target).unsqueeze(dim=0)
            target = F.resize(target, self.input_size, interpolation=InterpolationMode.NEAREST)
            target = torch.where(target > 0.5, 1., 0.)

        return image, target

    def __len__(self):
        """"
        データセットの長さを返す．len(dataset)のようにして取得可能．
        Returns: データセット内のデータ数
        """
        return len(self.image_files)


class PackDataset(torch.utils.data.Dataset):
    """
    PackageDatasetをロードするためのクラス
    """

    def __init__(self, root: str, category: str, valid_category: str, input_size: int, is_eval: bool = False, is_train: bool = True, test_ratio: float = 0.1, valid_ratio: float = 0.2,
                 is_mask: bool = False, patch_size: int = None, random_sampling: bool = False, seed: int = 42):
        """
        Args:
            root: PackageDatasetのルートディレクトリ．このディレクトリ直下に各クラスのデータディレクトリを含む．
            category: クラス名. Ex. 'cut'
            input_size: 事前学習済みモデルの入力画像の形．256と指定された場合は(256, 256)の画像を表す．
            is_eval: 評価かどうか.評価であればテストデータを利用．
            is_train: 訓練データかどうか
            test_ratio: 全体のデータのうち，何割を評価用に使うか
            valid_ratio: 評価データのうち何割を検証用に使うか(検証データはepoch数の設定などに利用)
            is_mask: 異常箇所のマスクがあるかどうか.
            patch_size: 画像をパッチに分割する際のパッチサイズ. input_size % patch_size = 0．パッチは重ならないように分割される．
            random_sampling: パッチ分割した画像をランダムに学習に用いる．ここで，パッチは重なっても良い．
            seed: 正常画像から訓練に使うものと評価に使うものを分割する際のランダムシード

        Attributes:
            self.image_transform: 画像に対するpreprocessing
            self.image_files: 画像のファイルパスのリスト
            self.label: 評価データにおける異常(1),正常(0)のラベル
            self.target_transform: 正解マスク画像に対するpreprocessing
        """
        self.input_size = input_size
        self.patch_size = patch_size

        # 事前学習済みモデルに入力するための画像変換を定義
        # ImageNetによる事前学習を想定しているため，ImageNet用の正規化をする．詳細は[https://teratail.com/questions/295871]を参照
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

            ]
        )

        # 正常・異常データのパスのリスト取得
        random.seed(seed)
        if category == 'all':
           data_dir = pathlib.Path(root)
           normal_list = list(data_dir.glob('*/OK_Clip/*.jpg'))
        else:
            normal_dir = pathlib.Path(os.path.join(root, category, 'OK_Clip'))
            normal_list = list(normal_dir.glob('*.jpg'))

        anomal_dir = pathlib.Path(os.path.join(root, valid_category, 'NG_Clip'))
        anomal_list = list(anomal_dir.glob('*.jpg'))

        # 正常画像をシャッフルし，test_ratioで指定された数の正常画像以外を訓練データとして利用
        random.shuffle(normal_list)
        random.shuffle(anomal_list)
        if is_train:
            self.image_files = normal_list[int(len(normal_list) * test_ratio):]
        else:
            # 正常画像の一部を評価用のデータとして利用
            self.image_files = normal_list[:int(len(normal_list) * test_ratio)]
            # ラベルを作成
            self.label = [0] * len(self.image_files)

            # 異常画像を評価用のデータとして利用
            self.image_files += anomal_list
            self.label += [1] * len(anomal_list)

            # 異常マスクがある場合には，マスクに対する変換も定義
            if is_mask:
                self.target_transform = transforms.Compose(
                    [
                        transforms.Resize(input_size),
                        # transforms.CenterCrop(input_size),
                        # transforms.ToTensor()
                    ]
                )
            # 検証データの数
            n_valid = int(len(self.label) * valid_ratio)
            eval_data_ = list(zip(self.image_files, self.label))
            random.shuffle(eval_data_)
            image_files, labels = zip(*eval_data_)
            # 検証時
            if not is_eval:
                self.image_files = list(image_files[:n_valid])
                self.label = list(labels[:n_valid])
            # 評価時
            else:
                self.image_files = list(image_files[n_valid:])
                self.label = list(labels[n_valid:])
                
        self.is_mask = is_mask
        self.is_train = is_train
        self.random = random_sampling

    def __getitem__(self, index):
        """
        データセットからindexで指定されたデータを取得する．dataset[idx]のようにリストから値を取得するように扱える．
        Args:
            index: データのindex

        Returns:
            output:
                if self.is_train: 事前学習済みモデルへの入力画像
                else: 事前学習済み画像への入力画像とtarget画像(異常箇所の正解マスク画像), マスクがない場合はラベル(0, 1)
        """
        # 指定されたindexの画像を読み込み，事前学習済みモデルに対応する形式に変換
        # -> (C, input_size, input_size)
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)

        if self.patch_size and not self.is_train:
            image = patch_split(image, self.patch_size)
        if self.patch_size and self.is_train:
            image = patch_split(image, self.patch_size, random_sampling=self.random)

        # 訓練時
        if self.is_train:
            return image

        # 評価時[マスクがない場合]
        if not self.is_mask:
            return image, self.label[index]

        # 評価時[マスクがある場合]
        # 正常画像に対する正解マスク画像の作成, targetは二値画像なので形は(1, input_size, input_size)
        if os.path.dirname(image_file).endswith("OK_Clip"):
            target = torch.zeros([1, *self.input_size])

        # 異常画像に対する正解マスク画像の作成と変換
        else:
            sep = os.path.sep
            target = Image.open(
                str(image_file).replace("NG_Clip", "NG_Clip_Label")
            )
            target = self.target_transform(target)
            target = (np.asarray(target, dtype=np.uint8) / 255)
            target = torch.from_numpy(target).unsqueeze(dim=0)
            target = torch.where(target > 0.5, 1., 0.)

        return image, target

    def __len__(self):
        """"
        データセットの長さを返す．len(dataset)のようにして取得可能．
        Returns: データセット内のデータ数
        """
        return len(self.image_files)


def build_train_data_loader(args, config: dict) -> torch.utils.data.DataLoader:
    """
    Args:
        args: ArgumentParserで受け取った引数を格納するNamespaceインスタンス
        config: 事前学習済みモデルの設定ファイル(yaml形式)から読み込んだ情報を格納する辞書

    Returns:
        訓練データのDataloader
    """

    # モデル固有の入力画像の形を取得 -> (H, W)
    input_size = literal_eval(config['input_size'])

    if args.name == 'mvtec':
        train_dataset = MVTecDataset(
            root=args.data,
            category=args.category,
            input_size=input_size,
            is_train=True,
            color=args.color
        )
    elif args.name == 'jelly':
        train_dataset = JellyDataset(
            root=args.data,
            category=args.category,
            valid_category=args.valid,
            input_size=input_size,
            is_train=True,
            is_mask=args.mask,
            patch_size=args.patchsize,
            random_sampling=args.random
        )
    elif args.name == 'package':
        train_dataset = PackDataset(
            root=args.data,
            category=args.category,
            valid_category=args.valid,
            input_size=input_size,
            is_eval=args.eval,
            is_train=True,
            is_mask=args.mask,
            patch_size=args.patchsize,
            random_sampling=args.random
        )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,  # TODO: it is True
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
    # モデル固有の入力画像の形を取得 -> (H, W)
    input_size = literal_eval(config['input_size'])

    if args.name == 'mvtec':
        test_dataset = MVTecDataset(
            root=args.data,
            category=args.category,
            input_size=input_size,
            is_train=False,
            is_mask=args.mask,
            color=args.color
        )
    elif args.name == 'jelly':
        test_dataset = JellyDataset(
            root=args.data,
            category=args.category,
            valid_category=args.valid,
            input_size=input_size,
            is_train=False,
            is_mask=args.mask,
            patch_size=args.patchsize
        )
    elif args.name == 'package':
        test_dataset = PackDataset(
            root=args.data,
            category=args.category,
            valid_category=args.valid,
            input_size=input_size,
            is_eval=args.eval,
            is_train=False,
            is_mask=args.mask,
            patch_size=args.patchsize,
            random_sampling=args.random
        )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def patch_split(img: torch.Tensor, patch_size: int, random_sampling: bool = False) -> torch.Tensor:
    """画像を複数のパッチに分割する．
    Args:
        img: 入力画像 (C, H, W), H=Wが前提
        patch_size: パッチサイズ
        random_sampling: ランダムにパッチをサンプリングする．この場合はパッチの重なりを許容する．

    Returns:
        patches: パッチに分割された画像 (N, C, P, P), N: パッチ数
    """

    # 入力のチェック
    assert img.shape[1] == img.shape[2], "入力画像の幅と高さは同じにしてください．"
    assert img.shape[1] % patch_size == 0, "画像サイズはパッチサイズで割りきれるようにしてください．"

    if random_sampling:
        step = const.STEP
    else:
        step = patch_size

    # パッチに分割
    # (C, H, W) -> (C, Nh, Nw, size, size)
    if random_sampling:
        patches = img.unfold(dimension=1, size=patch_size, step=step)
        patches = patches.unfold(dimension=2, size=patch_size, step=step)
    else:
        patches = img.unfold(dimension=1, size=patch_size, step=step)
        patches = patches.unfold(dimension=2, size=patch_size, step=step)

    # -> (Nh, Nw, C, size, size)
    patches = patches.permute([1, 2, 0, 3, 4])
    # -> (Nh*Nw, C, size, size)
    patches = patches.reshape(-1, 3, patch_size, patch_size)

    if random_sampling:
        rng = default_rng()
        sample_idx = rng.choice(patches.shape[0], size=const.NUM_PATCHES)
        patches = patches[sample_idx]

    return patches


if __name__ == '__main__':
    import tqdm
    dataset_dir = "C:\\Users\had-int22\PycharmProjects\Pytorch_AD\data\package\cut\\NG_Clip_Label"
    path_list = pathlib.Path(dataset_dir).iterdir()
    for path in tqdm.tqdm(path_list):
        parent = path.parent
        stem = path.stem
        new_path = parent / (stem + '.jpg')
        path.rename(new_path)