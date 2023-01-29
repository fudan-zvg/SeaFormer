# SeaFormer

## Usage

First, clone the repository locally:
```
git clone https://github.com/wwqq/Efficient-transformer.git
```
Then, install PyTorch 1.6.0+ and torchvision 0.7.0+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
pip install mmsegmentation
pip install mmcv
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Model Zoo

- SeaFormer on ImageNet-1K

| Method                    | Size | Acc@1 | #Params (M) | FLOPs (G) | Download |
|---------------------------|:----:|:-----:|:-----------:|:---------:|:--------:|
| SeaFormer-Tiny   |  224 |  67.9 |     1.8     |    0.1    |[[Google]]() [[Baidu]]() |
| SeaFormer-Small  |  224 |  73.1 |     4.1     |    0.2    |[[Google]]() [[Baidu]]() |
| SeaFormer-Base   |  224 |  75.9 |     8.7     |    0.3    |[[Google]]() [[Baidu]]() |
| SeaFormer-Large  |  224 |  79.4 |     14.0    |    1.2    |[[Google]]() [[Baidu]]() |

The password of Baidu Drive is seaf

## Evaluation
To evaluate a pre-trained SeaFormer-Tiny on ImageNet val with a single GPU run:
```
python validate.py /imagenet/validation/ --model SeaFormer_T --checkpoint /path/to/checkpoint_file
```


## Training
To train SeaFormer_Tiny on ImageNet on a single node with 8 gpus for 600 epochs run:

```
sh distributed_train.sh 8 /data-path -b 128 --model SeaFormer_T --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 7 --warmup-lr 1e-6 --weight-decay 2e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .064 --lr-noise 0.42 --output /output_dir --experiment SeaFormer_T
```


## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
