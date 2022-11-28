import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np

import epsAM as net
from sampler import InfiniteSamplerWrapper
import glob

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str,
                    default='Dataset/cocostuff/images/train2017')
parser.add_argument('--style_dir', type=str,
                    default='Dataset/PBN/train')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=2)
parser.add_argument('--save_model_interval', type=int, default=5000)

# custom training options
parser.add_argument('--from_pretrained', type=int, default=1)
parser.add_argument('--train_encoder', type=int, default=1)
parser.add_argument('--metric_learning', type=int, default=0)
parser.add_argument('--eps', type=float, default=1e-2)
args = parser.parse_args()

device = torch.device('cuda')
if args.eps <= 0:
    args.eps = 'diversify'
expt_id = f'epsAM-PT_{args.from_pretrained}-ENC_{args.train_encoder}-ML_{args.metric_learning}-EPS_{args.eps}'
save_dir = os.path.join(args.save_dir, expt_id)
save_dir = Path(save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(os.path.join(args.log_dir, expt_id))
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

# define model
encoder_trainable = True if args.train_encoder == 1 else False
metric_learning = True if args.metric_learning == 1 else False

decoder = net.get_decoder()
vgg = net.get_vgg()

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

# ============== load pretrained weights if choose ==========
if args.from_pretrained:
    decoder.load_state_dict(torch.load('models/decoder.pth'))

network = net.Net(vgg, decoder, encoder_trainable, metric_learning, args.eps)
network.train()
network.to(device)

vgg_for_loss_model = net.get_vgg()
vgg_for_loss_model.load_state_dict(torch.load(args.vgg))
loss_model = net.LossModel(vgg_for_loss_model)
loss_model.eval()
loss_model.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
if torch.cuda.device_count() > 1:
    network = torch.nn.DataParallel(network)
# ============== auto resume ==========
ckpts = sorted(glob.glob(os.path.join(save_dir, 'iter_*.pth.tar')))
if len(ckpts) > 0:
    import parse
    latest_ckpt = ckpts[-1]
    print(f'INFO: resume from {latest_ckpt}')
    ckpt = torch.load(latest_ckpt)
    network.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optim'])
    p = parse.compile("iter_{}.pth.tar")
    init_iter = int(p.parse(latest_ckpt.split('/')[-1])[0])
else:
    print('INFO: start from scratch')
    init_iter = 0

# ========= start train =============
for i in tqdm(range(init_iter, args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    try:
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)
    except:
        continue
    transferred_images = network(content_images, style_images)
    loss_c, loss_s = loss_model(content_images.detach(), style_images.detach(), transferred_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        torch.save(
            {
                'model': state_dict,
                'optim': optimizer.state_dict()
            }, 
            save_dir / 'iter_{:06d}.pth.tar'.format(i + 1)
        )
writer.close()
