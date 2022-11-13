import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', nargs='+', default=0, type=int)
args = parser.parse_args()

# setup gpu
args.gpus = args.gpus if isinstance(args.gpus, list) else [args.gpus]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in args.gpus])

import torch

print("device count:", torch.cuda.device_count())
print("available:", torch.cuda.is_available())

for c in range(torch.cuda.device_count()):
    dev = f'cuda:{c}'
    x = torch.randn((128, 1024, 64*(c + 1), 64*(c + 1))).to(dev)
    print("device:", dev, x.shape, x.device)
