"""
datasetを準備するモジュール．
Pytorchのtorch.utils.data APIを使用．詳細は公式チュートリアル[https://pytorch.org/tutorials/beginner/basics/data_tutorial.html]を参照．
"""

import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class MVTecDataset(torch.utils.data.Dataset):
    """
    MVTecDatasetをロードするためのクラス．
    """
    def __init__(self, root: str, category: str, input_size: int, is_train: bool = True):
        """
        Args:
            root: MVTecADのルートディレクトリ．このディレクトリ直下に各クラスのデータディレクトリを含む．
            category: クラス名. Ex. 'hazelnut'
            input_size: 事前学習済みモデルの入力画像の形．256と指定された場合は(256, 256)の画像を表す．
            is_train: 訓練データかどうか
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