import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets

class SubsetImageFolder(datasets.ImageFolder):

    def __init__(self,
                 root,
                 include_folders,
                 transform=None,
                 target_transform=None):

        assert len(include_folders) == len(set(include_folders))

        self.include_folders = include_folders
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform)

    def find_classes(self, directory):

        all_classes = [entry.name for entry in os.scandir(directory) if entry.is_dir()]

        if not all_classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        is_subset = set(self.include_folders).issubset(all_classes)
        if not is_subset:
            raise ValueError("Specified classes must be a subset of existing classes.")
        
        classes = sorted(cls for cls in all_classes if cls in self.include_folders)
        
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx 


def gen_datasets(datadir,
                 id_subset,
                 ood_subset,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 resize=256,
                 cropsize=224,
                 ):

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(cropsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = SubsetImageFolder(os.path.join(datadir, "train"),
                                      id_subset,
                                      train_transform)
    val_dataset = SubsetImageFolder(os.path.join(datadir, "val"),
                                    id_subset,
                                    eval_transform)
    ood_dataset = SubsetImageFolder(os.path.join(datadir, "val"),
                                    ood_subset,
                                    eval_transform)

    print("# ID Train: {}".format(len(train_dataset.imgs)))
    print("# ID Val: {}".format(len(val_dataset.imgs)))
    print("# OOD: {}".format(len(ood_dataset.imgs)))

              
    return train_dataset, val_dataset, ood_dataset


def gen_custom_dataset(datadir, name, subset, evaluate=True):

    # Imagenet stats
    cropsize = 224
    resize = 256
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(cropsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        normalize,
    ])

    transform = eval_transform if evaluate else train_transform

    dataset = SubsetImageFolder(os.path.join(datadir, name),
                                subset,
                                transform)

    print(f"# Samples {name}: {len(dataset.imgs)}")

    return dataset


def get_id_classes(id_classes_fn):

    with open(id_classes_fn, "r") as f:
        lines = [line.strip() for line in f]

    return sorted(lines)
