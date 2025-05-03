import argparse
import os
import types
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from lib.hierarchy import Hierarchy
from lib.utils.dataset_util import (gen_datasets,
                                    gen_custom_dataset,
                                    get_id_classes)

from lib.utils.hierarchy_utils import (get_multidepth_classes,
                                       get_multidepth_target_transform)

import torch.nn as nn
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument("--height", type=int, required=True)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--datadir", type=str, required=True)
parser.add_argument("--hierarchy", type=str, required=True)
parser.add_argument("--traindir", type=str, required=True)
parser.add_argument("--include_train", type=bool, default=False)
parser.add_argument("--extraood", type=str, default=None)
parser.add_argument("--id_split", type=str, required=True)

def main(args):

    batch_size = args.batch_size
    hierarchy_fn = args.hierarchy

    checkpoint_fn = os.path.join(args.traindir, "checkpoint.pt")

    id_classes = get_id_classes(args.id_split)

    hierarchy = Hierarchy(id_classes,
                          hierarchy_fn,
                          )

    ood_classes = hierarchy.ood_train_classes

    print("==> Preparing data..")
    train_ds, val_ds, ood_ds = gen_datasets(args.datadir,
                                            id_classes,
                                            ood_classes,
                                            )

    if args.extraood:
        oodextra_ds = gen_custom_dataset(args.datadir, args.extraood, ood_classes)
        ood_ds = torch.utils.data.ConcatDataset([oodextra_ds, ood_ds])

    height = args.height
    
    multi_classes = get_multidepth_classes(hierarchy, train_ds.classes)
    target_transform = get_multidepth_target_transform(train_ds.classes, multi_classes, height, hierarchy)
    
    num_id_classes = len(multi_classes[-(height + 1)])

    train_ds.target_transform = lambda x: target_transform[x]
    val_ds.target_transform = lambda x: target_transform[x]    

    trainloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=16)
    valloader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=16)
    oodloader = DataLoader(
        ood_ds, batch_size=batch_size, shuffle=False, num_workers=16)

    net = models.resnet50(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, num_id_classes)

    # override forward method to enable extracting features
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

        sets = [["val", valloader], ["ood", oodloader]]
        if args.include_train:
            sets.append(["train", trainloader])
        
        for dset, loader in sets:

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
                
            save_path = os.path.join(parent_dir, f"{dset}-preds.out")
            print(f"save_path: {save_path}")
            
            torch.save({"logits": logits, "targets": targets},
                       save_path)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
