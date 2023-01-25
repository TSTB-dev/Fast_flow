"""
DifferNetのモデル定義スクリプト. 公式実装[https://github.com/TSTB-dev/differnet/blob/master/model.py]を参考．
NormalizingFlowの実装にFrEIAというフレームワークを用いている．詳細はこちら[https://vislearn.github.io/FrEIA/_build/html/index.html]
timmは事前学習済みモデルを利用するためのライブラリ．詳細はこちら[https://github.com/rwightman/pytorch-image-models]
"""

from ast import literal_eval
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet

import constants as const


def subnet_fc_func(hidden_ratio: float):
    """Affine coupling layerにおけるsubnetworkを返す関数の定義
    Args:
        hidden_ratio: 隠れ層のニューロン数の拡張率
    """
    def subnet_fc(in_channels: int, out_channels: int):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(),
                             nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
                             nn.Linear(hidden_channels, out_channels))
    return subnet_fc


def nf_differnet(input_dim: int, hidden_ratio: float, flow_steps: int, clamp: float = 2.0):
    """Normalizing Flowを適用する．
    Args:
        input_dim: 入力される特徴ベクトルの長さ
        hidden_ratio: 特徴マップのチャンネル数の拡張率.
        flow_steps: Flowの数．
        clamp: スケーリングパラメータsの値域を指数関数に適用する前に，[-clamp, clamp]に制限

    Returns:
        nodes: Normalizing Flow全体のモデル．通常のtorch.nn.Moduleのように扱える．
    """
    nodes = Ff.SequenceINN(input_dim)
    for i in range(flow_steps):
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_fc_func(hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class DifferNet(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        flow_steps: int,
        input_size: int,
        feature_dim: int = 768,
        hidden_ratio: float = 1.0,
        n_scales: int = 3,
    ):
        """
        Args:
            backbone_name: 事前学習済みモデルの名称
            flow_steps: Flowの数
            input_size: 入力画像の幅と高さ. H=Wの正方形画像が前提．
            feature_dim: 特徴ベクトルの長さ
            hidden_ratio: 特徴マップのチャンネル数の拡張率.
        """
        super(DifferNet, self).__init__()
        if backbone_name == 'alexnet':
            self.feature_extractor = alexnet(weights='IMAGENET1K_V1')
        self.flow = nf_differnet(feature_dim, hidden_ratio, flow_steps)
        self.n_scales = n_scales
        self.img_size = input_size

        # 事前学習済みモデルの重みは凍結し，勾配も計算しない．
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):

        loss = 0
        y_cat = list()

        self.feature_extractor.eval()

        for s in range(self.n_scales):
            x_scaled = F.interpolate(x, size=self.img_size[0] // (2 ** s)) if s > 0 else x
            feat_s = self.feature_extractor.features(x_scaled)
            y_cat.append(torch.mean(feat_s, dim=(2, 3)))
        y = torch.cat(y_cat, dim=1)

        output, log_jac_dets = self.flow(y)
        # output: (B, C)
        loss += torch.mean(
            0.5 * torch.sum(output ** 2, dim=1) - log_jac_dets
        )
        ret = {'loss': loss}

        if not self.training:

            log_prob = -torch.mean(output ** 2, dim=1, keepdim=True) * 0.5
            probs = torch.exp(log_prob)
            ret = {'probs': probs}

        return ret


def build_model(config: dict, args) -> torch.nn.Module:
    """
    Args:
        config: 事前学習済みモデルの設定ファイル(yaml形式)から読み込んだ情報を格納する辞書
        args: ArgParserで受け取ったコマンドライン引数を格納するオブジェクト
    Returns:
        Model(FastFlow, DifferNet, ...etc)
    """
    # モデル固有の入力画像の形を取得 -> (H, W)
    input_size = literal_eval(config['input_size'])

    model = DifferNet(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=input_size,
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