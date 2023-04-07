import yaml
import os
import datetime
import pathlib

import torch
from torch.utils.tensorboard import SummaryWriter

import constants as const
import dataset
from evaluate import eval_once
import fastflow
import utils


def train_one_epoch(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> float:
    """1エポック訓練する．

    Args:
        dataloader: 訓練セットのデータローダ
        model: FastFlowのインスタンス
        optimizer: Optimizerのインスタンス
        epoch: 何エポック目か

    Returns:
        1エポックでのlossの平均
    """

    # 訓練モードに設定
    model.train()
    # 1epoch中におけるlossの平均を計算
    loss_meter = utils.AverageMeter()


    for step, data in enumerate(dataloader):
        # forward
        # data -> (img, cond), img: (batch_size, 3, 256, 256), cond: (batch_size, 1)
        img, cond = data[0].cuda(), data[1].cuda()
        ret = model(img, cond)
        loss = ret["loss"]

        # backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient norm clipping[https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update weights
        optimizer.step()

        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )

    return loss_meter.avg


def train(args):
    """モデルを訓練する
    Args:
        args: ArgumentParserで受け取った引数
    """
    init_epoch = 0

    # logを格納するディレクトリの作成とSummaryWriterの定義
    log_dir, start_time = utils.create_log_dir("fastflow", args.category)
    print(f"TensorBoard上で学習状況を確認するには，次のコマンドを実行してください．\n tensorboard --logdir={log_dir} --port 0")
    writer = SummaryWriter(log_dir)

    # checkpointを格納するディレクトリの作成
    save_dir = utils.create_save_dir('fastflow', args.category, args.data)
    ckpt_dir = os.path.join(save_dir, 'ckpt')
    pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # backboneのメタ情報を読み込み，モデルをビルド
    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    model = fastflow.build_model(config, args)
    optimizer = fastflow.build_optimizer(model)

    # ckptを指定した場合は読み込む．
    model.cuda()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        init_epoch = int(pathlib.Path(args.checkpoint).stem)

    # dataloaderを作成し，モデルをGPUに転送
    train_dataloader = dataset.build_train_data_loader(args, config)
    test_dataloader = dataset.build_test_data_loader(args, config)


    for epoch in range(init_epoch, init_epoch + const.NUM_EPOCHS):

        # パラメータを更新
        loss = train_one_epoch(train_dataloader, model, optimizer, epoch)
        writer.add_scalar('Loss/train', loss, epoch + 1)

        # 一定間隔ごとにテストデータ全てを使い，性能を評価(この性能によって訓練エポックを変えると，汎化性能が楽観的な評価になる)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            auroc = eval_once(args, test_dataloader, model, args.mask)
            writer.add_scalar('AUROC/test', auroc, epoch + 1)

        # 一定間隔ごとにモデルをsave
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(ckpt_dir, "%d.pt" % epoch),
            )

    end_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    utils.save_training_info(args, config, start_time, end_time, save_dir, log_dir, ckpt_dir)