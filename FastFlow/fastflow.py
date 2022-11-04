"""
FastFlowのモデル定義スクリプト.

NormalizinFlowの実装にFrEIAというフレームワークを用いている．詳細はこちら[https://vislearn.github.io/FrEIA/_build/html/index.html]
timmは事前学習済みモデルを利用するためのライブラリ．詳細はこちら[https://github.com/rwightman/pytorch-image-models]
"""

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as const


def subnet_conv_func(kernel_size: int, hidden_ratio: float):
    """指定されたカーネルサイズとカーネル数を持つ2層の畳み込み層を返す関数を返します．
    Args:
        kernel_size: 畳み込み層のカーネルサイズ
        hidden_ratio: 入力チャンネル数に対しての隠れ層のチャンネル数

    Returns:
        subnet_conv: 入力チャンネル数と出力チャンネル数を受け取って2層の畳み込み層を返す関数
    """
    def subnet_conv(in_channels: int, out_channels: int):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def nf_fast_flow(input_chw: list, conv3x3_only: bool, hidden_ratio: float, flow_steps: int, clamp: float = 2.0):
    """2D-Normalizing Flowを適用する．
    Args:
        input_chw: 入力される特徴マップの形: (C, H, W). C: Channel, H: Height, W: Width
        conv3x3_only: Flow内部の畳み込み層で，3x3の畳み込み層のみを使うかどうか．原論文ではFlow stepごとに1x1と3x3を交互に適用．
        hidden_ratio: 特徴マップのチャンネル数の拡張率.
        flow_steps: Flowの数．
        clamp: スケーリングパラメータsの値域を指数関数に適用する前に，[-clamp, clamp]に制限

    Returns:
        nodes: Normalizing Flow全体のモデル．通常のtorch.nn.Moduleのように扱える．
    """
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class FastFlow(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        flow_steps: int,
        input_size: int,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
    ):
        """
        Args:
            backbone_name: 事前学習済みモデルの名称
            flow_steps: Flowの数
            input_size: 入力画像の幅と高さ. H=Wの正方形画像が前提．
            conv3x3_only: Flow内部の畳み込み層で3x3のカーネルのみを使うかどうか．原論文ではFlow stepごとに1x1と3x3を交互に適用
            hidden_ratio: 特徴マップのチャンネル数の拡張率.
        """
        super(FastFlow, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "利用できる事前学習済みモデルは次の通りです： {}".format(const.SUPPORTED_BACKBONES)

        # 事前学習済みモデルの読み込み[ViT]
        if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [768]
            scales = [16]

        # 事前学習済みモデルの読み込み[ResNet], timm.create_modelの詳細は[https://rwightman.github.io/pytorch-image-models/feature_extraction/]
        else:
            # ResNetの場合，複数スケールの特徴マップを取得, FastFlow原論文の表7参照[https://arxiv.org/abs/2111.07677]
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )

            # channels: 各特徴マップのチャンネル数, scales: 各特徴マップの入力画像のスケール縮小率
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # LayerNormを定義
            # TransFormerについては後ろの方でLayerNormを適用．
            # ResNetに関しては複数の特徴マップがあるので，それぞれ個別に学習可能なLayerNormを適用．
            self.norms = nn.ModuleList()

            # 各特徴マップについてLayerNormを定義
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size / scale), int(input_size / scale)],
                        elementwise_affine=True,
                    )
                )

        # 事前学習済みモデルの重みは凍結し，勾配も計算しない．
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Flow部分の定義．特徴マップが複数ある場合はその個数分Flowを定義．
        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.input_size = input_size

    def forward(self, x):
        # 事前学習済みモデルを評価モードにする．BatchNormなど訓練時と評価時で挙動の変わる層があるため．
        self.feature_extractor.eval()

        # 事前学習済みモデルがDeiTの場合
        if isinstance(
            self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        ):
            # patchに分割し，パッチごとの埋め込みベクトルを獲得
            # (B, C, H, W) -> (B, N, D), N: パッチ数, D: 埋め込みベクトルの次元
            x = self.feature_extractor.patch_embed(x)

            # クラストークンのサイズをバッチサイズに合うように拡張
            # (1, 1, C) -> (B, 1, C)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)

            # dist_token(distillation token)がなければ，cls_tokenのみ結合．dist_tokenについてはDeiTの原論文参照[https://arxiv.org/abs/2012.12877]
            # dist_tokenがない場合，(B, N, D) -> (B, N+1, D). ある場合は(B, N, D) -> (B, N+2, D)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )

            # 位置埋め込み(Positional Encoding), 埋め込み後にdropoutを適用
            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
            for i in range(8):  # 原論文ではDeiTのBlockIndexは7, DeiTはDefault12ブロック
                x = self.feature_extractor.blocks[i](x)

            # LayerNormを適用
            # (B, N, D) -> (B, N, D)
            x = self.feature_extractor.norm(x)

            # トークンを抜き出す.(B, N+2, D) -> (B, N, D)
            x = x[:, 2:, :]
            B, _, D = x.shape

            # (B, N, D) -> (B, D, N)
            x = x.permute(0, 2, 1)

            # (B, D, N) -> (B, D, Np, Np). Np: 行/列方向のパッチの個数
            x = x.reshape(B, D, self.input_size // 16, self.input_size // 16)
            features = [x]

        # 事前学習済みモデルがCaiTの場合
        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            x = self.feature_extractor.patch_embed(x)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]

        # 事前学習済みモデルがResNetの場合
        else:
            # 特徴マップの抽出（M個のスケールの特徴マップを抽出）
            # (B, C, H, W) -> (B, M, D, H', W'). M: 特徴マップの数.
            features = self.feature_extractor(x)

            # 各スケールの特徴マップについてLayerNorm
            features = [self.norms[i](feature) for i, feature in enumerate(features)]

        loss = 0
        outputs = []

        for i, feature in enumerate(features):
            # 潜在変数z_0と対数ヤコビ行列式を取得
            # output: (B, in_channels, H', W'), in_channels:入力チャンネル数, H', W':特徴マップの幅と高さ
            # log_jac_dets: (B, )
            output, log_jac_dets = self.nf_flows[i](feature)
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
            )
            outputs.append(output)
        ret = {"loss": loss}

        # 評価時(model.eval()で評価モードに設定すると，training=Falseになる)
        if not self.training:
            anomaly_map_list = []
            for output in outputs:
                log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
                prob = torch.exp(log_prob)
                a_map = F.interpolate(
                    -prob,
                    size=[self.input_size, self.input_size],
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_map_list.append(a_map)
            anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
            anomaly_map = torch.mean(anomaly_map_list, dim=-1)
            ret["anomaly_map"] = anomaly_map

        return ret