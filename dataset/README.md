# Dataset

1. Referring segmentation datasets (**Required for both training and eval**): [(FP-/R-)refcoco(+/g) annotations](https://drive.google.com/file/d/1mA3kcY3QiAZz1Zr89MCKYd7e3LBIwUzl/view?usp=sharing), [COCO images](http://images.cocodataset.org/zips/train2014.zip)

2. Visual Question Answering dataset (**Required for training models for referring segmentation model**): [LLaVA-Instruct-150k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json)

1. Semantic segmentation datasets (**Required for training models for reasoning segmentation tasks**): [ADE20K](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip), [COCO-Stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip), [PACO-LVIS](https://github.com/facebookresearch/paco/tree/main#dataset-setup), [PASCAL-Part](https://github.com/facebookresearch/VLPart/tree/main/datasets#pascal-part), [COCO Images](http://images.cocodataset.org/zips/train2017.zip)

    Note: For COCO-Stuff, we use the annotation file stuffthingmaps_trainval2017.zip. We only use the PACO-LVIS part in PACO. COCO Images should be put into the `dataset/coco/` directory.

5. Augmented Reasoning segmentation dataset (with false-premise queries): [FP-Aug ReasonSeg](https://drive.google.com/file/d/11WNg1KaV2mk7gTdJRa2aahGqfj4luTDw/view?usp=sharing)

Download them from the above links, and organize them as follows.

```
SESAME
├── dataset
│   ├── ade20k
│   │   ├── annotations
│   │   └── images
│   ├── coco
│   │   └── train2017
│   │       ├── 000000000009.jpg
│   │       └── ...
│   ├── cocostuff
│   │   └── train2017
│   │       ├── 000000000009.png
│   │       └── ...
│   ├── llava_dataset
│   │   └── llava_instruct_150k.json
│   ├── reason_seg
│   │   └── ReasonSeg
│   │       ├── train
│   │       └── val
│   ├── refer_seg
│   │   ├── images
│   │   |   └── mscoco
│   │   |       └── images
│   │   |           └── train2014
│   │   ├── refclef
│   │   ├── refcoco
│   │   ├── refcoco+
│   │   ├── refcocog
│   │   ├── R-refcoco
│   │   ├── R-refcoco+
│   │   ├── R-refcocog
│   │   ├── fprefcoco
│   │   ├── fprefcoco+
│   │   └── fprefcocog
│   ├── vlpart
│   │   ├── paco
│   │   │   └── annotations
│   │   └── pascal_part
│   │       ├── train.json
│   │       └── VOCdevkit
│   ├── clip-vit-large-patch14-336
│   ├── llava-v1.5-7b
│   ├── llava-v1.5-13b
│   ├── SESAME-LLaVA-v1.5-7B
│   └── sam_vit_h_4b8939.pth
```
