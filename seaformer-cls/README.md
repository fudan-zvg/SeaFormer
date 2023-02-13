# SeaFormer

## Usage

First, clone the repository locally:
```
git clone https://github.com/fudan-zvg/SeaFormer
cd SeaFormer/seaformer-cls
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
| SeaFormer-Tiny   |  224 |  68.1 |     1.8     |    0.1    |[[Google]](https://drive.google.com/file/d/1Qh1-8n-KUSGCoKVVcvjuHkTb3AWcX7o5/view?usp=sharing) [[Baidu]](https://pan.baidu.com/s/1Kr6XEZUTHUZkZcAfSswMEw) |
| SeaFormer-Small  |  224 |  73.4 |     4.1     |    0.2    |[[Google]](https://drive.google.com/file/d/1_jpChZPVDFbHfxZPQFL3wKPuDcDgEtPK/view?usp=share_link) [[Baidu]](https://pan.baidu.com/s/1LrySouv0BzRwqvi4D7vCzA) |
| SeaFormer-Base   |  224 |  76.4 |     8.7     |    0.3    |[[Google]](https://drive.google.com/file/d/1pTqkYXmEfGuRD2119vWikwd68lGNo8Lx/view?usp=sharing) [[Baidu]](https://pan.baidu.com/s/1jXQmZKzneLy3g4GX8aaibQ) |
| SeaFormer-Large  |  224 |  79.9 |     14.0    |    1.2    |[[Google]](https://drive.google.com/file/d/1FYyCHV1deCs02ims2Y-NybAMlo_a5o2p/view?usp=sharing) [[Baidu]](https://pan.baidu.com/s/12ZA6L4lcCLo3nEMidAI-wA) |

The password of Baidu Drive is seaf

## Evaluation
To evaluate a pre-trained SeaFormer-Tiny on ImageNet val with a single GPU run:
```
python validate.py /imagenet/validation/ --model SeaFormer_T --checkpoint /path/to/checkpoint_file
```


## Training
To train SeaFormer_Tiny on ImageNet on a single node with 8 gpus for 600 epochs run:

```
sh distributed_train.sh 8 /data-path -b 128 --model SeaFormer_T --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 7 --warmup-lr 1e-6 --warmup-epochs 10 --weight-decay 2e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --mixup 0.0 --amp --lr .064 --lr-noise 0.42 --output /output_dir --experiment SeaFormer_T --resume /resume_dir
```

To train SeaFormer_Small on ImageNet on a single node with 8 gpus for 600 epochs run:
```
sh distributed_train.sh 8 /data-path -b 128 --model SeaFormer_S --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 7 --warmup-lr 1e-6 --warmup-epochs 40 --weight-decay 2e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --mixup 0.2 --amp --lr .064 --lr-noise 0.42 --output /output_dir --experiment SeaFormer_S --resume /resume_dir
```

To train SeaFormer_Base on ImageNet on a single node with 8 gpus for 600 epochs run:
```
sh distributed_train.sh 8 /data-path -b 128 --model SeaFormer_B --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 7 --warmup-lr 1e-6 --warmup-epochs 10 --weight-decay 2e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --mixup 0.2 --amp --lr .064 --lr-noise 0.42 --output /output_dir --experiment SeaFormer_B --resume /resume_dir
```

To train SeaFormer_Large on ImageNet on a single node with 8 gpus for 600 epochs run:
```
sh distributed_train.sh 8 /data-path -b 128 --model SeaFormer_L --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 7 --warmup-lr 1e-6 --warmup-epochs 10 --weight-decay 2e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --mixup 0.2 --amp --lr .064 --lr-noise 0.42 --output /output_dir --experiment SeaFormer_L --resume /resume_dir
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
