# LC-MSM

This repository contains the official code for <br>
**"LC-MSM: Language-Conditioned Masked Segmentation Model for Unsupervised Domain Adaptation"**

 ## Model Performance
 ||GTA to Cityscapes|SYNTHIA to Cityscapes|
 |---|---|---|
 |DAFormer|68.3|60.9|
 |LC-MSM (Single-resolution)|71.8|62.8|
 |HRDA|73.8|65.8|
 |LC-MSM (Multi-resolution)|76.0|68.2|

 ![performance](imgs/performance.png)

## Usage

### environments
- Ubuntu 20.04
- Python 3.8.5

### Requirements
- torch >= 1.8.0
- torchvision
- mmcv-full
- open-clip
- tqdm

To use this code, please first install the 'mmcv-full' by following the official guideline guidelines ([`mmcv`](https://github.com/open-mmlab/mmcv/blob/master/README.md)).


The requirements can be installed by the following command
```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install open_clip_torch
```

### Prepare dataset
**Citysacpes**: Please, download leftImg8bit_trainvaltest.zip and trainvaltest.zip from [`here`](https://www.cityscapes-dataset.com/)

<br>

**GTA**: Please download all images and label packages from [`here`](https://download.visinf.tu-darmstadt.de/data/from_games/)

<br>

**SYNTHIA**: Please download SYNTHIA-RAND-CITYSCAPES from [`here`](https://synthia-dataset.net/downloads/)

### Pre-trained backbone weight
please download the pre-trained weight for MiT-B5 via shell script
```shell
sh tools/download_checkpoints.sh
```

### Preprocessing the dataset
The following dataset is preprocessed in COCO format , but if you are using the raw json file you can preprocess with the script
```shell
python tools/convert_datasets/gta.py /your/path/
python tools/convert_datasets/cityscapes.py /your/path/
python tools/convert_datasets/synthia.py /your/path/
```

### Training
For convenience, provides and [annotated config file](configs/daformer/gta2cs_uda_pseudoclip_context.py) of the adaptation model<br>
**Before training the data path in the [datset config file](configs/_base_/datasets/uda_gta_to_cityscapes_512x512.py) should be modifed with your data path.**


A training job can be launched using:

```shell
python run_experiment.py --config configs/daformer/gta2cs_uda_pseudoclip_context.py
```

### Evaluating
The checkpoint will be sevaed automatically in work_dirs, else you set a directory for it.

```shell
sh test.sh path/to/checkpoint/directory
```
