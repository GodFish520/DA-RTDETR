# DA-RTDETR
## Acknowledgment
This implementation is bulit upon [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
## Installation
Please refer to the instructions [here](requirements.txt). We leave our system information for reference.

* OS: Ubuntu 22.04.4
* Python: 3.11
* CUDA: 11.8
* PyTorch: 2.0.1 (You can only use this version.)
* torchvision: 0.15.2

## Dataset Preparation
Please construct the datasets following these steps:

- Download the datasets from their sources.

- Convert the annotation files into COCO-format annotations.

- The datasets download link is here: [cityscapes and foggy_cityscapes](https://www.cityscapes-dataset.com/) [KITTI](https://www.cvlibs.net/datasets/kitti/) [BDD100k](https://bair.berkeley.edu/blog/2018/05/30/bdd/)
```
  #---Source Domain
    PATHS_Source = {
        "train": ("",  #train image dir
                  ""), #train coco format json file
        "val": ("",    #val image dir
                ""),   #val coco format json file
    }
    #----Target Domain
    PATHS_Target = {
        "train": ("",  #train image dir
                  ""), #train coco format json file
        "val": ("",    #val image dir
                ""),   #val coco format json file
    }
```
You need to modify the configs file before training, for example the data dir and the oouput dir

## Training
We provide training script as follows.
- Training with single GPU
```
python train.py -c configs/rtdetr/r50_city.yml --use_pixel_da --use_instance_da
```
- Training with Multi-GPU
```
torchrun --nproc_per_node=2 train.py -c configs/rtdetr/r50_city.yml --use_pixel_da --use_instance_da
```

## Reference
https://github.com/lyuwenyu/RT-DETR