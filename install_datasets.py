import os
from pathlib import Path
from itertools import chain
from tqdm import tqdm
from PIL import Image
import argparse
import re
import csv


def gen_imagenet_val_classdirs(imagenet_source, imagenet_dest, solution_csv_path):
    """
    imagenet_source: source directory of imagenet dataset
    imagenet_dest: root imagenet directory at destination
    solution_csv_path: full path to csv file containing labels for val files
    """

    dest_val_path = os.path.join(imagenet_dest, "val")
    source_val_path = os.path.join(imagenet_source, "val")
    
    os.makedirs(dest_val_path, exist_ok=True)

    with open(solution_csv_path, "r") as f:

        reader = csv.reader(f)
        next(reader) # skip the header
        
        for line in tqdm(reader, desc="Creating symlinks for ImageNet val files"):
            match_ = re.search(r"n[0-9]{8}", line[1])

            if match_:
                ncode = match_.group(0)
                img = line[0].split(",")[0] # TODO: this works but does not seem necessary
                dest_classdir = os.path.join(dest_val_path, ncode)
                os.makedirs(dest_classdir, exist_ok=True)

                source = os.path.join(source_val_path, f"{img}.JPEG")
                dest = os.path.join(dest_classdir, f"{img}.JPEG")

                try:
                    os.symlink(source, dest)
                except FileExistsError:
                    pass


def create_symlinks(source_dir, dest_dir):
    """
    Generates symlinks in dest_dir to all files in source_dir.
    """

    # Ensure the source directory exists
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    for filename in tqdm(os.listdir(source_dir),
                         desc=f"Creating symlinks for {dest_dir}"):
        source_path = os.path.join(source_dir, filename)

        # Only process files, not subdirectories
        if os.path.isfile(source_path):
            dest_path = os.path.join(dest_dir, filename)

            try:
                os.symlink(source_path, dest_path)
                # print(f"Symlink created: {dest_path} -> {source_path}")
            except OSError as e:
                print(f"Error creating symlink for {source_path}: {e}")


def create_inat_symlinks(file_list_path, source_dir, dest_dir):
    """
    file_list_path: full path to file containing filenames
    source_dir: source inat directory
    dest_dir: destination inat directory
    """

    # Ensure the source directory exists
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")

    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Ensure the file list exists
    if not os.path.isfile(file_list_path):
        raise FileNotFoundError(f"File list '{file_list_path}' does not exist.")

    with open(file_list_path, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]

    # find class directory
    file_list = os.path.basename(file_list_path)
    inat_number = ''.join(char for char in file_list if char.isdigit())
    inat_integer = int(inat_number)

    dirs = [list(p.iterdir()) for p in Path(source_dir).iterdir() if p.is_dir()]
    dirs = list(chain.from_iterable(dirs))

    matching_paths = [path for path in dirs if path.name == str(inat_integer)]
    assert len(matching_paths) == 1
    source_classdir = matching_paths[0]
    
    for filename in filenames:

        dest_class_name = file_list[:-4]
        source_path = os.path.join(source_dir, source_classdir, filename)
        dest_path = os.path.join(dest_dir, dest_class_name, filename)

        os.makedirs(os.path.join(dest_dir, dest_class_name), exist_ok=True)

        try:
            # skip if already exists
            if os.path.exists(dest_path):
                print(f"{dest_path} already exists")
                continue
            
            # Create the symlink
            os.symlink(source_path, dest_path)
            
        except OSError as e:
            print(f"Error creating symlink for {source_path}: {e}")

def preprocess_fgvc(source_dir):

    image_dir = os.path.join(source_dir, "data", "images")
    annotation_dir = os.path.join(source_dir, "data")

    train_dir = os.path.join(source_dir, "train")
    val_dir = os.path.join(source_dir, "val")
    test_dir = os.path.join(source_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    def process_annotation_line(line):
        line = line.strip()
        tokens = line.split(" ")
        img_name = tokens[0]
        label = "_".join(tokens[1:])

        # replace / with + to enable class names as directory names
        label = label.replace("/","+")

        return img_name, label

    def process_file(img_name, img_label, dest_dir):
        img_label = f"v-{img_label}"
        src_path = os.path.join(image_dir, f"{img_name}.jpg")
        label_dir = os.path.join(dest_dir, img_label)
        os.makedirs(label_dir, exist_ok=True)
        dest_path = os.path.join(label_dir, f"{img_name}.jpg")

        if os.path.isfile(dest_path):
            return
        
        img = Image.open(src_path)
        col, row = img.size
        img = img.crop((0, 0, col, row - 20)) # crop out bottom banner
        img.save(dest_path)
        
    with open(os.path.join(annotation_dir, "images_variant_train.txt")) as f:
        train_labels = list(f)

    for line in tqdm(train_labels, desc="Preprocessing train files"):
        img_name, img_label = process_annotation_line(line)
        process_file(img_name, img_label, train_dir)

    with open(os.path.join(annotation_dir, "images_variant_val.txt")) as f:
        val_labels = list(f)

    for line in tqdm(val_labels, desc="Preprocessing val files"):
        img_name, img_label = process_annotation_line(line)
        process_file(img_name, img_label, val_dir)

    with open(os.path.join(annotation_dir, "images_variant_test.txt")) as f:
        test_labels = list(f)

    for line in tqdm(test_labels, desc="Preprocessing test files"):
        img_name, img_label = process_annotation_line(line)
        process_file(img_name, img_label, test_dir)


def install_fgvc(source_dir, dest_dir):
    """
    source_dir: source fgvc-aircraft directory
    dest_dir: destination data directory (an fgvc-aircraft directory will be created inside this directory)
    """

    name = "fgvc-aircraft"

    print(f"Installing {name}...")

    preprocess_fgvc(source_dir)
    
    source_train = os.path.join(source_dir, "train")
    source_val = os.path.join(source_dir, "val")
    source_test = os.path.join(source_dir, "test")

    dest_train = os.path.join(dest_dir, name, "train")
    dest_val = os.path.join(dest_dir, name, "val")

    # train -> train
    for class_ in os.listdir(source_train):
        class_source = os.path.join(source_train, class_)
        if os.path.isdir(class_source):
            class_dest = os.path.join(dest_train, class_)
            create_symlinks(class_source, class_dest)

    # val -> train
    for class_ in os.listdir(source_val):
        class_source = os.path.join(source_val, class_)
        if os.path.isdir(class_source):
            class_dest = os.path.join(dest_train, class_)
            create_symlinks(class_source, class_dest)

    # test -> val
    for class_ in os.listdir(source_test):
        class_source = os.path.join(source_test, class_)
        if os.path.isdir(class_source):
            class_dest = os.path.join(dest_val, class_)
            create_symlinks(class_source, class_dest)


def install_inat(source_dir, dest_dir, splits_dir):
    """
    source_dir: source inat directory
    dest_dir: destination data directory
    splits_dir: directory containing files for train/val split
    """
    
    name = "inat19"

    print(f"Installing {name}...")

    train_split = os.path.join(splits_dir, "train")
    val_split = os.path.join(splits_dir, "val")

    target_inat = os.path.join(dest_dir, name)

    os.makedirs(target_inat, exist_ok=True)

    for f in tqdm(os.listdir(train_split), desc="Creating symlinks for training set"):
        file_path = os.path.join(train_split, f)
        dest = os.path.join(target_inat, "train")
        create_inat_symlinks(file_path, source_dir, dest)

    for f in tqdm(os.listdir(val_split), desc="Creating symlinks for validation set"):
        file_path = os.path.join(val_split, f)
        dest = os.path.join(target_inat, "val")
        create_inat_symlinks(file_path, source_dir, dest)

def install_imagenet(source_dir, dest_dir, solution_csv_path):
    """
    source_dir: source imagenet directory
    dest_dir: destination data directory
    solution_csv_path: path to csv file containing labels for val files
    """

    name = "imagenet"

    print(f"Installing {name}...")
    
    train_source = os.path.join(source_dir, "train")
    
    dest_imagenet = os.path.join(dest_dir, name)
    os.makedirs(dest_imagenet, exist_ok=True)

    dest_train = os.path.join(dest_imagenet, "train")

    if not os.path.exists(dest_train):
        os.symlink(train_source, dest_train)

    gen_imagenet_val_classdirs(source_dir, dest_imagenet, solution_csv_path)


def main(args):

    if args.inat_source:
        assert args.inat_splits
        install_inat(args.inat_source, args.destdir, args.inat_splits)

    if args.fgvc_source:
        install_fgvc(args.fgvc_source, args.destdir)

    if args.imagenet_source:
        assert args.imagenet_vallabels
        install_imagenet(args.imagenet_source, args.destdir, args.imagenet_vallabels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create symlinks for a dataset structure.")
    parser.add_argument("--fgvc_source", type=str, help="Source directory for FGVC-Aircraft")
    parser.add_argument("--inat_source", type=str, help="Source directory for iNaturalist")
    parser.add_argument("--inat_splits", type=str, help="Directory containing iNaturalist split files")
    parser.add_argument("--imagenet_source", type=str, help="Source directory for ImageNet")
    parser.add_argument("--imagenet_vallabels", type=str, help="Path to CSV file for ImageNet val labels")
    parser.add_argument("--destdir", type=str, help="Path to the destination data directory.")

    args = parser.parse_args()

    main(args)
