import os
import sys
import time
import argparse

from tqdm import tqdm

import torch
import torch.nn as nn

from config import HPARAMS_REGISTRY, setup_data

from utils.logger import Logger
from utils.helpers import timer, count_parameters, extract_hparams
from utils.helpers import load_checkpoint, save_checkpoint

from hip import HierarchicalPerceiver, ConvHierarchicalPerceiver

torch.autograd.set_detect_anomaly(True)


parser = argparse.ArgumentParser(description="PyTorch Hierarchical Perceiver")
parser.add_argument('--config', '-c', default='cifar10_learned', type=str, metavar='NAME',
                    help='Config for HiP (default: cifar10_learned)')
parser.add_argument('--name', '-n', default='', type=str, metavar='NAME',
                    help='Run name and folder where logs are stored')
parser.add_argument('--gpus', default=0, type=int, nargs='+', metavar='GPUS',
                    help='If GPU(s) available, which GPU(s) to use for training.')
parser.add_argument('--pos', default=None, metavar='POS', type=str,
                    help='Positional embedding. Default is None, which keeps the predefined setting')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='Number of epochs to run (default: 2)')
parser.add_argument('--lr', default=0.0001, type=float, metavar='LR',
                    help='Initial learning rate (default: 0.0001)')
parser.add_argument('--weight-decay', default=0.05, type=float, metavar='WD',
                    help='Weight decay for optimizer (default: 0.05)')
parser.add_argument('--batch-size', default=None, metavar='N', type=int,
                    help='Mini-batch size')
parser.add_argument('--ckpt-save', default=False, action=argparse.BooleanOptionalAction, dest='save_checkpoint',
                    help='Save checkpoints to folder')
parser.add_argument('--load-ckpt', default=None, metavar='PATH', dest='load_checkpoint',
                    help='Load model checkpoint and train/evaluate.')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='Evaluate model on test set')
parser.add_argument('--log-save-interval', default=5, type=int, metavar='N', dest='save_interval',
                    help="Interval in which logs are saved to disk (default: 5)")
parser.add_argument('--debug', action='store_true',
                    help="If set, saves all logs and models in a specific debug folder")


LOG_DIR = 'runs'
logger = Logger(LOG_DIR)


def main():
    global logger
    args = parser.parse_args()
    H = HPARAMS_REGISTRY[args.config]

    # overwrite settings if required
    if args.batch_size is not None:
        H.batch_size = args.batch_size
    else:
        args.batch_size = H.batch_size

    if args.pos is not None:
        H.pos_embedding = args.pos
    else:
        args.pos = H.pos_embedding

    # GPU
    args.gpus = args.gpus if isinstance(args.gpus, list) else [args.gpus]
    if len(args.gpus) == 1:
        device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    else:
        raise ValueError('Currently multi-gpu training is not possible')

    # data
    data = setup_data(H)

    # model
    if H.conv:
        model = ConvHierarchicalPerceiver(H)
    else:
        model = HierarchicalPerceiver(H)
    model.to(device)

    # loss function and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)

    if args.load_checkpoint:
        model, start_epoch = load_checkpoint(model, args.load_checkpoint, device)
        args.epochs += start_epoch
    else:
        start_epoch = 0

    if args.evaluate:
        logger = Logger(tensorboard=False, create_folder=False)
        logger.init_epoch()
        validate(model, data.val, criterion, device)
        print("\n{:>8}: {:.4f} - {:>8}: {:.4f}".format('val_loss', logger.epoch['val_loss'].avg,
                                                       'val_acc', logger.epoch['val_acc'].avg))
        return

    # setup paths and logging
    debug_dir = 'debug' if args.debug else ''
    running_log_dir = os.path.join(LOG_DIR, debug_dir, H.dataset, H.name, H.pos_embedding, args.name)
    logger = Logger(running_log_dir, tensorboard=True)
    logger.log_hparams(extract_hparams(H, args, model))

    # output ######################
    for name, val in vars(args).items():
        print("{:<16}: {}".format(name, val))
    print("{:<16}: {}".format('device', device))
    print("{:<16}: {}".format('embedding', H.pos_embedding))
    print("{:<16}: {}".format('2d patches', H.patch_2d))
    print("{:<16}: {}".format('logdir', running_log_dir))
    print("{:<16}: {}".format('params', count_parameters(model)))
    ###############################

    # epoch run ###################
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        logger.init_epoch(epoch)
        print(f"Epoch [{epoch + 1} / {args.epochs}]")

        train(model, data.train, criterion, optimizer, device, H)
        validate(model, data.val, criterion, device, H)

        # output progress
        print(f"{'loss':>8}: {logger.epoch['loss'].avg:.4f} - {'val_loss':>8}: {logger.epoch['val_loss'].avg:.4f} - "
              f"{'acc':>4}: {logger.epoch['acc'].avg:.4f} - {'val_acc':>4}: {logger.epoch['val_acc'].avg:.4f}")

        # save logs and checkpoint
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            logger.save()
            if args.save_checkpoint:
                ckpt_dir = os.path.join(running_log_dir, 'checkpoints')
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir, exist_ok=True)
                save_checkpoint(model, ckpt_dir, logger)

        logger.end_epoch()
    ###############################

    logger.tensorboard.add_hparams(logger.hparams,
                                   {'train_loss': min(logger.summary['loss']),
                                    'val_loss': min(logger.summary['val_loss']),
                                    'train_acc': max(logger.summary['acc']),
                                    'val_acc': max(logger.summary['val_acc'])},
                                   run_name='.')

    elapsed_time = timer(t_start, time.time())
    print(f"Total training time: {elapsed_time}")


def train(model, train_loader, criterion, optimizer, device, hps):
    model.train()
    for x, y in tqdm(train_loader, desc="Training"):
        x, y = x.to(device), y.to(device)

        y_hat = model(x)

        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = torch.argmax(y_hat, dim=1).eq(y).sum()
        acc = correct / y.shape[0]

        logger.log_metrics({'loss': loss, 'acc': acc},
                           phase='train', aggregate=True, n=x.shape[0])

        if hps.step_log_interval is not None and logger.global_train_step % hps.step_log_interval == 0:
            logger.tensorboard.add_scalar('Accuracy/train', logger.epoch['acc'].avg, logger.global_train_step)
            logger.tensorboard.add_scalar('Loss/train', logger.epoch['loss'].avg, logger.global_train_step)

        logger.global_train_step += 1
    log_step = logger.global_train_step if hps.step_log_interval is not None else logger.running_epoch
    logger.tensorboard.add_scalar('Accuracy/train', logger.epoch['acc'].avg, log_step)
    logger.tensorboard.add_scalar('Loss/train', logger.epoch['loss'].avg, log_step)


@torch.no_grad()
def validate(model, val_loader, criterion, device, hps):
    model.eval()
    for x, y in tqdm(val_loader, desc="Validation"):
        x, y = x.to(device), y.to(device)

        y_hat = model(x)

        loss = criterion(y_hat, y)

        correct = torch.argmax(y_hat, dim=1).eq(y).sum()
        acc = correct / y.shape[0]
        logger.log_metrics({'val_loss': loss, 'val_acc': acc},
                           phase='val', aggregate=True, n=x.shape[0])

    log_step = logger.global_train_step if hps.step_log_interval is not None else logger.running_epoch
    logger.tensorboard.add_scalar('Accuracy/val', logger.epoch['val_acc'].avg, log_step)
    logger.tensorboard.add_scalar('Loss/val', logger.epoch['val_loss'].avg, log_step)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exit training with keyboard interrupt!")
        logger.save()
        sys.exit(0)
