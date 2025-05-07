# ProHOC

This is the official repository for the CVPR 2025 paper:
**[ProHOC: Probabilistic Hierarchical Out-of-Distribution Classification via Multi-Depth Networks](https://arxiv.org/abs/2503.21397)**.

If you have questions or would like to discuss the paper or code, don't hesitate to get in touch!

## Requirements

Python dependencies are listed in [requirements.txt](./requirements.txt).

Install them using:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Manually download the following datasets:

* [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
* [iNaturalist 2019](https://github.com/visipedia/inat_comp/tree/master/2019)
* [ImageNet 2012](https://www.image-net.org/challenges/LSVRC/2012/index.php)

Extract the datasets into directories of your choice.


## Set Environment Variables

Before running the installation script, set the following environment variables:

```bash

# Path to the ProHOC repository
PROHOC=/path/to/prohoc/

# Paths to raw datasets
FGVCSOURCE=/path/to/fgvc-aircraft-2013b/
IMAGENETSOURCE=/path/to/imagenet/
INATSOURCE=/path/to/train_val2019/

# Path to iNat train-val splits (provided in this repo)
INATSPLITS=$PROHOC/splits_inat19

# ImageNet validation labels
IMAGENETVALLABELS=/path/to/imagenet/LOC_val_solution.csv

# Output paths
PROHOCDATA=/path/to/prohoc-data/       # Where processed data/symlinks will go
TRAINDIR=/path/to/prohoc-experiments/  # Where results/checkpoints will be saved
```

## Install Datasets

Run the installation script:

```bash
python3 $PROHOC/install_datasets.py \
  --fgvc_source $FGVCSOURCE \
  --imagenet_source $IMAGENETSOURCE \
  --inat_source $INATSOURCE \
  --inat_splits $INATSPLITS \
  --imagenet_vallabels $IMAGENETVALLABELS \
  --destdir $PROHOCDATA
```

*To skip installing a dataset, simply omit its source argument.*

## Training

Train the multi-depth networks for each dataset and hierarchy depth.

### iNaturalist19

```bash
HEIGHTS=(0 1 2 3 4 5)
DSET=inat19

for HEIGHT in "${HEIGHTS[@]}"; do
  python3 $PROHOC/main_multidepth.py \
    --datadir $PROHOCDATA/$DSET/ \
    --hierarchy $PROHOC/hierarchies/$DSET.pth \
    --traindir $TRAINDIR/$DSET/H$HEIGHT \
    --id_split $PROHOC/data/$DSET-id-labels.csv \
    --height $HEIGHT \
    --epochs 90 \
    --lr 0.01
done
```

### FGVC-Aircraft

```bash
HEIGHTS=(0 1 2)
DSET=fgvc-aircraft

for HEIGHT in "${HEIGHTS[@]}"; do
  python3 $PROHOC/main_multidepth.py \
    --datadir $PROHOCDATA/$DSET/ \
    --hierarchy $PROHOC/hierarchies/$DSET.pth \
    --traindir $TRAINDIR/$DSET/H$HEIGHT \
    --id_split $PROHOC/data/$DSET-id-labels.csv \
    --height $HEIGHT \
    --epochs 90 \
    --lr 0.01
done
```

### SimpleHierImagenet

```bash
HEIGHTS=(0 1 2 3 4 5 6 7 8 9 10)
DSET=simple-hier-imagenet
DATASOURCE=imagenet

for HEIGHT in "${HEIGHTS[@]}"; do
  python3 $PROHOC/main_multidepth.py \
    --datadir $PROHOCDATA/$DATASOURCE/ \
    --hierarchy $PROHOC/hierarchies/$DSET.json \
    --traindir $TRAINDIR/$DSET/H$HEIGHT \
    --id_split $PROHOC/data/$DSET-id-labels.csv \
    --height $HEIGHT \
    --epochs 150 \
    --lr 0.05
done
```

**Note:** Although the hierarchy is named simple-hier-imagenet, it is
built using a class subset from the ImageNet dataset. Therefore, the
data directory (--datadir) points to imagenet, not
simple-hier-imagenet.

*Training jobs can also be run in parallel if your system supports it.*

## Generate Predictions

Generate validation logits for ID and OOD data to enable evaluating
the hierarchical inference.

### FGVC-Aircraft

```bash
DSET=fgvc-aircraft
HEIGHTS=(0 1 2)

for HEIGHT in "${HEIGHTS[@]}"; do
  python3 $PROHOC/gather_vallogits_multidepth.py \
    --datadir $PROHOC/$DSET/ \
    --traindir $TRAINDIR/$DSET/H$HEIGHT \
    --height $HEIGHT \
    --id_split $PROHOC/data/$DSET-id-labels.csv \
    --hierarchy $PROHOC/hierarchies/$DSET.json
done
```

### iNaturalist19

```bash
DSET=inat19
HEIGHTS=(0 1 2 3 4 5)

for HEIGHT in "${HEIGHTS[@]}"; do
  python3 $PROHOC/gather_vallogits_multidepth.py \
    --datadir $PROHOC/$DSET/ \
    --traindir $TRAINDIR/$DSET/H$HEIGHT \
    --height $HEIGHT \
    --id_split $PROHOC/data/$DSET-id-labels.csv \
    --hierarchy $PROHOC/hierarchies/$DSET.json
done
```

### SimpleHierImagenet

```bash
DSET=simple-hier-imagenet
HEIGHTS=(0 1 2 3 4 5 6 7 8 9 10)

for HEIGHT in "${HEIGHTS[@]}"; do
  python3 $PROHOC/gather_vallogits_multidepth.py \
    --datadir $PROHOC/$DSET/ \
    --traindir $TRAINDIR/$DSET/H$HEIGHT \
    --height $HEIGHT \
    --id_split $PROHOC/data/$DSET-id-labels.csv \
    --hierarchy $PROHOC/hierarchies/$DSET.json
done
```

---

## Evaluate ProHOC

Evaluate hierarchical predictions:

```bash
DSET=inat19

python3 $PROHOC/gather_hinference.py \
  --basedir $TRAINDIR/$DSET/ \
  --uncertainty_methods compprob entcompprob \
  --id_split $PROHOC/data/$DSET-id-labels.csv \
  --hierarchy $PROHOC/hierarchies/$DSET.json
```

*Results will be saved to:*

```
$TRAINDIR/$DSET/hinference-inat19.result
```

You can load and inspect the results using:

```python
import torch
torch.load(...)
```

## Hierarchies

The class hierarchies are defined in JSON files located in the
[hierarchies](https://github.com/walline/prohoc/tree/main/hierarchies)
directory. Each JSON file represents a tree-structured class
hierarchy, where every node corresponds to a class in the dataset.

Each node in the JSON files contains the following fields:

- `name`: A string identifying the class. This should match the corresponding class directory name of the dataset.
- `description`: A human-readable description of the class, useful for clarifying non-descriptive class names (such as WordNet IDs).
- `children`: A list of child nodes, each following the same structure, representing subclasses. This list is empty for leaf nodes.

To define a new hierarchy, create a JSON file with the same structure. The top-level node needs to be named "root".

## Notes

The ID/OOD splits of the datasets are specified in the csv files
contained in `$PROHOC/data/`. Other ID/OOD splits can be used by
creating new files with other classes.

Out-of-hierarchy datasets can be evaluated by first generating their
predictions with `$PROHOC/gather_faroodlogits_multidepth` and then
including them when running `$PROHOC/gather_hinference.py` using the
`--farood *MY-DATASET*` flag with `--betas 1`.

## Acknowledgement

Parts of this repository are adapted from [rwl93/hierarchical-ood](https://github.com/rwl93/hierarchical-ood).
Thanks to the authors for their valuable contributions.
