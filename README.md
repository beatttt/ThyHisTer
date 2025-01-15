# ThyHisTer

This repo is the suppliments of paper "ThyHisTer: a new thyroid histopathology image dataset for ternary
classification of thyroid cancer", including code for training and test, along with partial data as demo.

To obtain the whole dataset, e-mail tzczsq@163.com.

## Dependencies

To install dependencies, run:

```bash
pip3 install -r requirements.txt
```

## Dataset

The original data type of this dataset is `.svs`, which is widely used by various of medical scanners. One SVS file consists of billions of pixels. But for this repo, PNG images was used for training. To crop a high-resolution SVS image into many relatively low-resolution PNG images, [CLAM](https://github.com/mahmoodlab/CLAM) was used here.

After cropping, PNG images should be arranged as:

```filestructure
ThyHisTer
|-train
| |-BN_1
| | |-n_BN_1_0.png
| | |-t_BN_1_1.png
| | |-...
| |-PTC_1
| |-FTC_1
| |-...
|-validation
|-evaluation
```

Filename rules:
* BN_1: Name of the SVS slide, from which all PNG in this directory were cropped. This dataset includes three classes: BN(Benign), PTC(Papillary Thyroid Carcinoma) and FTC(Follicular Thyroid Carcinoma).
* n_BN_1_0.png: n/t is the label for classification, which represents Normal or Tumor. BN_1 indicates the PNG image was cropped from SVS image BN_1. 0 is the index number.

## Finetune

To finetune a pretrained model, download pretrained weights, and put it in directory `pretrained_weights`. Then run:

```bash
python src\train.py --model-name densenet161 --dataset E:\patch\ --save-path ..\finetuned_weights --batch-size 512 --data-parallel
```

## Evaluation

To evaluate a model, run:
```bash
python src\slide_test.py --model-name resnet18 --dataset E:\patch\validation\ --batch-size 2048 --data-parallel
```

The Evaluation results will be stored in directory `evaluation_results`.

## Help

To get help for command, run:

```bash
python src\train.py -h
```

or

```bash
python src\slide_test.py -h
```