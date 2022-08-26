# CDR-BIQA
The codes for the work "Many Heads Are Better than One: Cross-Dataset-Robust Method for Blind Real-World Images Quality Assessment".

## Dependencies

```
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1
```

- Install `timm==0.3.2`:

```
pip install timm==0.3.2
```

- Install `Apex`:

```
https://github.com/NVIDIA/apex
cd apex
python setup.py install
```

- Install other requirements:

```
pip install opencv-python yacs openpyxl scipy numpy pillow termcolor
```


## Usages

### Testing a single image

Predicting image quality with our model CDR-BIQA [Online](http://online-ip.xyz).

To run the demo, please download the pre-trained model at [Google drive](https://drive.google.com/file/d/1g75-Wf7F21l5dfU3WTVo2PJAyiwZuXSZZ/view?usp=sharing), put it in [pretrained_models](pretrained_models) folder, then run:

```
python demo.py
```

### Training
#### SwinT-IQA
Training individual SwinT-IQA model on each BIQA dataset

```
python train_swintiqa.py
```

Some available options in config_swintiqa.py:
* `_C.DATA.DATASET`: Training and testing dataset, support datasets: [livec](https://live.ece.utexas.edu/research/ChallengeDB/index.html) | [koniq-10k](http://database.mmsp-kn.de/koniq-10k-database.html) | [bid](https://drive.google.com/drive/folders/1Qmtp-Fo1iiQiyf-9uRUpO-YAAM0mcIey) | [spaq](https://github.com/h4nwei/SPAQ) | [flive](https://github.com/baidut/paq2piq).
* `_C.DATA.TRAIN_PATCH_NUMBER`: Sampled image patch number per training image.
* `_C.DATA.TEST_PATCH_NUMBER`: Sampled image patch number per testing image.
* `_C.DATA.BATCH_SIZE`: Batch size.
* `_C.MODEL.TYPE`: Backbone, support model: [swin](https://github.com/microsoft/Swin-Transformer) | resnet | [convnext](https://github.com/facebookresearch/ConvNeXt)

#### CDR-BIQA
- Prepare data: please download the datasets at [Google drive](https://drive.google.com/drive/folders/18qnWf7NEDokkfeBkCXoxttFZMp9Tfa5l?usp=sharing).
- Generate pseudo-labels: please run [labels_with_sigmoid.m](data\generate_pseudolabels) to calculate relative probability pseudo-labels. The scores from individual SwinT-IQA model trained on each BIQA dataset.

- Train
```
python train_cdrbiqa.py
```
- Test
```
python test_cdrbiqa.py
```
## References
* [hyperIQA](https://github.com/SSL92/hyperIQA)
* [UNIQUE](https://github.com/zwx8981/UNIQUE)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)


