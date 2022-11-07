"""
datasetを準備するモジュール．
Pytorchのtorch.utils.data APIを使用．詳細は公式チュートリアル[https://pytorch.org/tutorials/beginner/basics/data_tutorial.html]を参照．
"""

import os
from glob import glob
import pathlib
import random

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class MVTecDataset(torch.utils.data.Dataset):
    """
    MVTecDatasetをロードするためのクラス．
    """

    def __init__(self, root: str, category: str, input_size: int, is_train: bool = True, is_mask: bool = False):
        """
        Args:
            root: MVTecADのルートディレクトリ．このディレクトリ直下に各クラスのデータディレクトリを含む．
            category: クラス名. Ex. 'hazelnut'
            input_size: 事前学習済みモデルの入力画像の形．256と指定された場合は(256, 256)の画像を表す．
            is_train: 訓練データかどうか
            is_mask: 異常箇所のマスクがあるかどうか．
        """

        # 事前学習済みモデルに入力するための画像変換を定義
        # ImageNetによる事前学習を想定しているため，ImageNet用の正規化をする．詳細は[https://teratail.com/questions/295871]を参照
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
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

    def __init__(self, root: str, category: str, input_size: int, is_train: bool = True, test_ratio: float = 0.2,
                 is_mask: bool = False, seed: int = 42):
        """
        Args:
            root: JellyDatasetのルートディレクトリ．このディレクトリ直下に各クラスのデータディレクトリを含む．
            category: クラス名. Ex. 'hair1'
            input_size: 事前学習済みモデルの入力画像の形．256と指定された場合は(256, 256)の画像を表す．
            is_train: 訓練データかどうか
            test_ratio: 全体のデータのうち，何割を評価用に使うか
            is_mask: 異常箇所のマスクがあるかどうか．
            seed: 正常画像から訓練に使うものと評価に使うものを分割する際のランダムシード

        Attributes:
            self.image_transform: 画像に対するpreprocessing
            self.image_files: 画像のファイルパスのリスト
            self.label: 評価データにおける異常(1),正常(0)のラベル
            self.target_transform: 正解マスク画像に対するpreprocessing
        """

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
        normal_dir = pathlib.Path(os.path.join(root, category, 'OK_Clip'))
        anormal_dir = pathlib.Path(os.path.join(root, category, 'NG_Clip'))
        normal_list = list(normal_dir.glob('*.jpg'))
        anormal_list = list(anormal_dir.glob('*.jpg'))

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
            self.image_files += anormal_list
            self.label += [1] * len(anormal_list)

            # 異常マスクがある場合には，マスクに対する変換も定義
            if is_mask:
                self.target_transform = transforms.Compose(
                    [
                        transforms.Resize(input_size),
                        transforms.ToTensor(),
                    ]
                )
        self.is_train = is_train

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

        if self.is_train:
            return image

        else:
            label = self.label[index]
            return image, label

    def __len__(self):
        """"
        データセットの長さを返す．len(dataset)のようにして取得可能．
        Returns: データセット内のデータ数
        """
        return len(self.image_files)
