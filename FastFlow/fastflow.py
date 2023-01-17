"""
FastFlowのモデル定義スクリプト.

NormalizingFlowの実装にFrEIAというフレームワークを用いている．詳細はこちら[https://vislearn.github.io/FrEIA/_build/html/index.html]
timmは事前学習済みモデルを利用するためのライブラリ．詳細はこちら[https://github.com/rwightman/pytorch-image-models]
"""
DIMS_OUT = 128
COND_DIMS = (32, )

from ast import literal_eval

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as const


class ConditioningNetwork(nn.Module):
    def __init__(self, dims_in: int, neurons: int, dims_out: int):
        super(ConditioningNetwork, self).__init__()
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(dims_in, neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons, dims_out)
        )

    def forward(self, c: torch.Tensor):
        return self.ffn(c)


def subnet_conv_func(kernel_size: int, hidden_ratio: float):
    """指定されたカーネルサイズとカーネル数を持つ2層の畳み込み層を返す関数を返す．
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


def nf_fast_flow(input_chw: list, conv3x3_only: bool, hidden_ratio: float, flow_steps: int, clamp: float = 2.0, feature_size=[]):
    """2D-Normalizing Flowを適用する．
    Args:
        input_chw: 入力される特徴マップの形: (C, H, W). C: Channel, H: Height, W: Width
        conv3x3_only: Flow内部の畳み込み層で，3x3の畳み込み層のみを使うかどうか．原論文ではFlow stepごとに1x1と3x3を交互に適用．
        hidden_ratio: 特徴マップのチャンネル数の拡張率.
        flow_steps: Flowの数．
        clamp: スケーリングパラメータsの値域を指数関数に適用する前に，[-clamp, clamp]に制限
        feature_size: 入力特徴マップのサイズ

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
            # TODO: conditional input
            cond=0,
            cond_shape=tuple(feature_size),
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
        patch_size: int = None,
        random_sampling: bool = False,
    ):
        """
        Args:
            backbone_name: 事前学習済みモデルの名称
            flow_steps: Flowの数
            input_size: 入力画像の幅と高さ. H=Wの正方形画像が前提．
            conv3x3_only: Flow内部の畳み込み層で3x3のカーネルのみを使う．原論文ではFlow stepごとに1x1と3x3を交互に適用
            hidden_ratio: 特徴マップのチャンネル数の拡張率.
            patch_size: パッチに分割されている場合のパッチサイズ
            random_sampling: パッチをランダムに選択して学習する．
        """
        super(FastFlow, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "利用できる事前学習済みモデルは次の通りです： {}".format(const.SUPPORTED_BACKBONES)

        # パッチに分割されている場合は入力画像のサイズをパッチサイズに変更
        if patch_size:
            input_size = (patch_size, patch_size)
        self.patch_size = patch_size

        # 事前学習済みモデルの読み込み[ViT]
        if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT, const.BACKBONE_DEIT_224, const.BACKBONE_DEITS_224, const.BACKBONE_MOBILEVIT_V2]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [self.feature_extractor.num_features]  # [self.feature_extractor.feature_info[-1]['num_chs']]
            scales = [16]


        # 事前学習済みモデルの読み込み[ResNet], timm.create_modelの詳細は[https://rwightman.github.io/pytorch-image-models/feature_extraction/]
        else:
            # ResNetの場合，複数スケールの特徴マップをlistで取得, FastFlow原論文の表7参照[https://arxiv.org/abs/2111.07677]
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )

            # channels: 各特徴マップのチャンネル数, scales: 各特徴マップの入力画像のスケール縮小率
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()
            self.feature_size = np.split(np.array([size // scale for scale in scales for size in input_size]), 3)

            # LayerNormを定義
            # TransFormerについては後ろの方でLayerNormを適用．
            # ResNetに関しては複数の特徴マップがあるので，それぞれ個別に学習可能なLayerNormを適用．
            self.norms = nn.ModuleList()
            self.cond_norms = nn.ModuleList()
            for scale in scales:
                self.cond_norms.append(
                    nn.LayerNorm(
                        [DIMS_OUT, int(input_size[0] / scale), int(input_size[1] / scale)],
                        elementwise_affine=True,
                    )
                )

            # 各特徴マップについてLayerNormを定義
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size[0] / scale), int(input_size[1] / scale)],
                        elementwise_affine=True,
                    )
                )

        # 事前学習済みモデルの重みは凍結し，勾配も計算しない．
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Flow部分の定義．特徴マップが複数ある場合はその個数分Flowを定義．
        # TODO: チャンネル数を修正
        self.nf_flows = nn.ModuleList()
        '''
        self.nf_flows.append(
            nf_fast_flow(
                [1792, 120, 160],
                conv3x3_only=conv3x3_only,
                hidden_ratio=hidden_ratio,
                flow_steps=flow_steps,
            )
        )
        '''
        self.cfnn = ConditioningNetwork(dims_in=1792, neurons=512, dims_out=DIMS_OUT)
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size[0] / scale), int(input_size[1] / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                    feature_size=[DIMS_OUT, input_size[0]//scale, input_size[1]//scale],
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
            if self.patch_size:
                # パッチに分割し，パッチごとの埋め込みベクトルを獲得
                # (B, N_g, C, P, P) -> (B, N_g, D, P, P), N_g: 画像全体をパッチに分割したときのパッチ数, D: 埋め込みベクトルの次元, P: ViT内部でのパッチ分割処理におけるパッチサイズ
                features = []
                for i in range(x.shape[1]):
                    feature = self.feature_extractor.patch_embed(x[:, i])

                    # クラストークンのサイズをバッチサイズに合うように用意
                    # (1, 1, D) -> (B, 1, D)
                    cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)

                    # dist_token(distillation token)がなければ，cls_tokenのみ結合．dist_tokenについてはDeiTの原論文参照[https://arxiv.org/abs/2012.12877]
                    # dist_tokenがない場合，(B, N_l, D) -> (B, N_l+1, D). ある場合は(B, N_l, D) -> (B, N_l+2, D)
                    if self.feature_extractor.dist_token is None:
                        feature = torch.cat((cls_token, feature), dim=1)
                    else:
                        dist_token = self.feature_extractor.dist_token.expand(x.shape[0], -1, -1)
                        feature = torch.cat((cls_token, feature, dist_token), dim=1)

                    # 位置埋め込み(Positional Encoding), 埋め込み後にdropoutを適用
                    feature = self.feature_extractor.pos_drop(feature + self.feature_extractor.pos_embed)
                    for i in range(8):  # 原論文ではDeiTのBlockIndexは7, DeiTはDefault12ブロック
                        feature = self.feature_extractor.blocks[i](feature)

                    # LayerNormを適用
                    # (B, N_g, N_l, D) -> (B, N_g, N_l, D)
                    feature = self.feature_extractor.norm(feature)

                    # トークンを抜き出す.(B, N+2, D) -> (B, N, D)
                    feature = feature[:, 2:, :]
                    B, _, D = feature.shape

                    # (B, N, D) -> (B, D, N)
                    feature = feature.permute(0, 2, 1)

                    # (B, D, N) -> (B, D, Np, Np). Np: 行/列方向のパッチの個数
                    feature = feature.reshape(B, D, self.input_size[0] // 16, self.input_size[1] // 16)
                    features.append(feature)
                features = [torch.stack(features, dim=0).transpose(0, 1)]

            else:
                # パッチに分割し，パッチごとの埋め込みベクトルを獲得
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
                x = x.reshape(B, D, self.input_size[0] // 16, self.input_size[1] // 16)
                features = [x]

        # MobileViTの場合
        elif isinstance(self.feature_extractor, timm.models.byobnet.ByobNet):
            # パッチに分割し，パッチごとの埋め込みベクトルを獲得
            # (B, N_g, C, P, P) -> (B, N_g, D, P, P), N_g: 画像全体をパッチに分割したときのパッチ数, D: 埋め込みベクトルの次元, P: ViT内部でのパッチ分割処理におけるパッチサイズ
            features = []
            if self.patch_size:
                for i in range(x.shape[1]):
                    # -> (B, D, P, P)
                    feature = self.feature_extractor.forward_features(x[:, i])
                    features.append(feature)
                features = [torch.stack(features, dim=0).transpose(0, 1)]
            else:
                feature = self.feature_extractor.forward_features(x)
                features = [feature]

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
            x = x.reshape(N, C, self.input_size[0] // 16, self.input_size[1] // 16)
            features = [x]

        # 事前学習済みモデルがResNetの場合
        else:

            if self.patch_size:
                # 各パッチについてM個のスケールの特徴マップを抽出
                # (B, N, C, P, P) -> (M], B, N, D, P, P), M]はサイズMのリストを表す
                feature_list = []
                for i in range(x.shape[1]):
                    features = self.feature_extractor(x[:, i])  # features: (M], D, )
                    features = [self.norms[i](feature) for i, feature in enumerate(features)]
                    feature_list.append(features)
                # feature_list: (N], M], B, N, D, P, P)
                # 転置して，(M], N, B, D, P, P)とする. [https://note.nkmk.me/python-list-transpose/]
                feature_list = [list(x) for x in zip(*feature_list)]
                feature_list = [torch.stack(x, dim=0) for x in feature_list]
                # -> (M], B, N, D, P, P)
                features = [x.transpose(0, 1) for x in feature_list]


            else:
                # 特徴マップの抽出（M個のスケールの特徴マップを抽出）
                # (B, C, H, W) -> (B, M, D, H', W'). M: 特徴マップの数, D: 各特徴マップのチャンネル数(WRN50_2の場合，256, 512, 1024)
                features = self.feature_extractor(x)
                conditions = [torch.nn.AvgPool2d(kernel_size=feature.shape[-2:])(feature) for feature in features]
                conditions = torch.concat(conditions, dim=1)[..., 0, 0]
                conditions = self.cfnn(conditions)

                # 各スケールの特徴マップについてLayerNorm
                # -> (M], B, D, H', W')
                features = [self.norms[i](feature) for i, feature in enumerate(features)]
                # concat
                # -> (1], B, D, H', W')
                '''
                max_shape = features[0].shape[-2:]
                features = [
                    features[0],
                    F.interpolate(features[1], size=max_shape, mode='bilinear'),
                    F.interpolate(features[2], size=max_shape, mode='bilinear')
                ]
                features = [torch.cat(features, dim=1)]
                '''

        loss = 0
        outputs = []

        # 各スケールの特徴マップを個別にNormalizingFlowにかける
        for i, feature in enumerate(features):

            if self.patch_size:
                # 各パッチの潜在変数z_0と対数ヤコビ行列式を取得
                # feature: (B, N, D, P, P)
                # output: (B, N, D, P, P)
                # log_jac_dets: (B, N)
                # outputs: (M], N], D, P, P)
                output_patch = []
                for j in range(feature.shape[1]):
                    output, log_jac_dets = self.nf_flows[i](feature[:, j])
                    loss += torch.mean(
                        0.5 * torch.sum(output ** 2, dim=(1, 2, 3)) - log_jac_dets
                    )
                    output_patch.append(output)
                outputs.append(output_patch)

            else:
                # 潜在変数z_0と対数ヤコビ行列式を取得
                # output: (B, in_channels, H', W'), in_channels:入力チャンネル数, H', W':特徴マップの幅と高さ
                # log_jac_dets: (B, )

                # (B, COND_DIM)
                c = torch.unsqueeze(torch.unsqueeze(conditions, dim=1), dim=1)
                c = torch.repeat_interleave(torch.repeat_interleave(c, self.feature_size[i][0], dim=1), self.feature_size[i][1], dim=2)
                c = torch.permute(c, [0, 3, 1, 2])

                c = self.cond_norms[i](c)

                # TODO: conditional inputs
                output, log_jac_dets = self.nf_flows[i](feature, c=[c])
                loss += torch.mean(
                    0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
                )
                outputs.append(output)
        ret = {"loss": loss}

        # 評価時(model.eval()で評価モードに設定すると，training=Falseになる)
        if not self.training:
            anomaly_map_list = []
            for output in outputs:
                if self.patch_size:
                    anomaly_maps_patch = []
                    for patch in output:
                        # patch: (B, D, P, P)
                        # 最後の潜在変数z0の分布として多変量標準ガウス分布を仮定している．各潜在変数ベクトルはこの分布からサンプリングされたものなので，確率を計算できる．
                        # 以下の処理は潜在変数ベクトルがわかっているときに，そこから確率を求める処理．詳細は多変量標準ガウス分布の公式を参照．
                        log_prob = -torch.mean(patch**2, dim=1, keepdim=True) * 0.5
                        prob = torch.exp(log_prob)
                        # -> (B, 1, P, P)
                        a_map = F.interpolate(
                            -prob,
                            size=[self.input_size[0], self.input_size[1]],
                            mode="bilinear",
                            align_corners=False,
                        )
                        anomaly_maps_patch.append(a_map)
                    anomaly_map_list.append(anomaly_maps_patch)  # -> (M], N], B, 1, P, P)

                else:
                    log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
                    prob = torch.exp(log_prob)
                    a_map = F.interpolate(
                        -prob,
                        size=[self.input_size[0], self.input_size[1]],
                        mode="bilinear",
                        align_corners=False,
                    )
                    anomaly_map_list.append(a_map)  # -> (M], B, 1, P, P)

            # 通常: -> (B, 1, P, P, M), パッチ分割時: -> (N], B, 1, P, P, M)
            if self.patch_size:
                anomaly_map_list = [torch.stack(x, dim=0) for x in anomaly_map_list]  # -> (B, N, 1, P, P, M)
                anomaly_map_list = torch.stack(anomaly_map_list, dim=-1).transpose(0, 1)
            else:
                anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)

            # 各スケールの異常マップを平均
            # 通常: -> (B, 1, P, P), パッチ分割時: -> (B, N, 1, P, P)
            anomaly_map = torch.mean(anomaly_map_list, dim=-1)
            ret["anomaly_map"] = anomaly_map

        return ret


def build_model(config: dict, args) -> torch.nn.Module:
    """
    Args:
        config: 事前学習済みモデルの設定ファイル(yaml形式)から読み込んだ情報を格納する辞書
        args: ArgParserで受け取ったコマンドライン引数を格納するオブジェクト
    Returns:
        FastFlowのインスタンス
    """
    # モデル固有の入力画像の形を取得 -> (H, W)
    input_size = literal_eval(config['input_size'])

    model = FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=input_size,
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
        patch_size=args.patchsize,
        random_sampling=args.random,
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