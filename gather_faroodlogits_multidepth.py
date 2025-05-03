import argparse
import os
import types
from pathlib import Path
import random

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from lib.hierarchy import Hierarchy
from lib.utils.hierarchy_utils import get_multidepth_classes
from lib.utils.dataset_util import get_id_classes

import torch.nn as nn
import torchvision.models as models

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument("--height", type=int, required=True)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--datadir", type=str, required=True)
parser.add_argument("--oodsets", nargs="+", type=str, required=True)
parser.add_argument("--traindir", type=str, required=True)
parser.add_argument("--hierarchy", type=str, required=True)
parser.add_argument("--id_split", type=str, required=True)

def main(args):

    batch_size = args.batch_size

    checkpoint_fn = os.path.join(args.traindir, "checkpoint.pt")

    id_classes = get_id_classes(args.id_split)

    hierarchy = Hierarchy(id_classes,
                          args.hierarchy)

    multi_classes = get_multidepth_classes(hierarchy, id_classes)

    num_id_classes = len(multi_classes[-(args.height + 1)])

    cropsize = 224
    resize = 256
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        normalize,
    ])

    loaders = []
    for ood_set in args.oodsets:
        data_root = os.path.join(args.datadir, ood_set)
        ds = ImageFolder(data_root, transform=transform)

        # TODO: make this random subsampling an argument
        n = 10000

        if len(ds) > n:
            random.seed(1234)
            indices = random.sample(range(len(ds)), n)
            ds = Subset(ds, indices)

        loader = DataLoader(ds, batch_size=batch_size, num_workers=16)
        loaders.append(loader)

    net = models.resnet50(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, num_id_classes)

    def forward(self, x: Tensor, embed=False) -> Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if embed:
            return self.fc(x), x

        x = self.fc(x)

        return x

    net.forward = types.MethodType(forward, net)

    net = net.to(device)
    
    print(f"checkpoint_fn: {checkpoint_fn}")
    
    # Load checkpoint
    net.load_state_dict(torch.load(checkpoint_fn))
    net.eval()

    print("Generating Results")
    with torch.no_grad():

        for dset, loader in zip(args.oodsets, loaders):

            print(f"Working on {dset}...")
            logits = torch.empty((0,), device="cpu")
            targets = torch.empty((0,), dtype=torch.long, device="cpu")
            #feats = torch.empty((0,), device="cpu")
            
            for inputs, targs in loader:
                inputs = inputs.to(device)
                outputs, f = net(inputs, embed=True)
                logits = torch.cat((logits, outputs.detach().cpu()), 0)
                targets = torch.cat((targets, targs.long()), 0)
                #feats = torch.cat((feats, f.detach().cpu()), 0)

            parent_dir = Path(checkpoint_fn).parent

            fn = f"{dset}-preds.out"
            fn = fn.replace("/", "-")
            save_path = os.path.join(parent_dir, fn)
            print(f"save_path: {save_path}")
            
            torch.save({"logits": logits, "targets": targets},
                       save_path)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
