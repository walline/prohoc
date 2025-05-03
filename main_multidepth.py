import argparse
import random
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os

from lib import train_util
from lib.hierarchy import Hierarchy
from lib.utils.dataset_util import gen_datasets, get_id_classes
from lib.utils.hierarchy_utils import get_multidepth_classes, get_multidepth_target_transform
import torchvision.models as models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--height", type=int, required=True)
parser.add_argument("--epochs", type=int, default=90)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.90)
parser.add_argument("--lr_decay", type=float, default=0.5)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--nesterov", type=bool, default=False)
parser.add_argument("--datadir", type=str, required=True)
parser.add_argument("--hierarchy", type=str, required=True)
parser.add_argument("--traindir", type=str, required=True)
parser.add_argument("--seed", type=int, default=123456)
parser.add_argument("--loginterval", type=int, default=250)
parser.add_argument("--id_split", type=str, required=True)


def main(args):

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    id_classes = get_id_classes(args.id_split)

    hierarchy = Hierarchy(id_classes,
                          args.hierarchy,
                          )

    ood_classes = hierarchy.ood_train_classes

    print("==> Preparing data..")
    train_ds, val_ds, _ = gen_datasets(args.datadir,
                                       id_classes,
                                       ood_classes,
                                       )

    height = args.height

    assert height < hierarchy._max_depth

    multi_classes = get_multidepth_classes(hierarchy, train_ds.classes)

    target_transform = get_multidepth_target_transform(train_ds.classes,
                                                       multi_classes,
                                                       args.height,
                                                       hierarchy)

    num_id_classes = len(multi_classes[-(height + 1)])

    print(f"Height: {height}")
    print(f"Nr classes: {num_id_classes}")
    
    train_ds.target_transform = lambda x: target_transform[x]
    val_ds.target_transform = lambda x: target_transform[x]
    
    trainloader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=16)
    valloader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=16)

    checkpoint_fn = os.path.join(args.traindir, "checkpoint.pt")
    print(f"checkpoint filename: {checkpoint_fn}")

    tensorboard_dir = os.path.join(args.traindir, "tensorboard")
    summary_writer = SummaryWriter(tensorboard_dir)

    pretrained = False if "imagenet" in args.datadir else True

    net = models.resnet50(pretrained=pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_id_classes)
    
    criterion = nn.CrossEntropyLoss()

    print(net)
    print(f"Device: {DEVICE}")
    net = net.to(DEVICE)

    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)
    
    top1_acc, top5_acc = train_util.train(
        net,
        trainloader,
        valloader,
        criterion,
        optimizer,
        args.epochs,
        args.batch_size,
        num_id_classes,
        checkpoint=checkpoint_fn,
        log_every_n=args.loginterval,
        summary_writer=summary_writer,
    )

    print(f"Top 1 Accuracy: {top1_acc}")
    print(f"Top 5 Accuracy: {top5_acc}")


if __name__=="__main__":
    args = parser.parse_args()
    main(args)
