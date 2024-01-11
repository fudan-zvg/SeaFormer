# SeaFormer


## Requirements

- pytorch 1.5+
- mmcv-full==1.3.14


## Main results
Results on ADE20K
Model | Params | FLOPs | mIoU(ss)    | Link
--- |:---:|:---:|:---:|:---: |
SeaFormer-T_512x512_2x8_160k | 1.7 | 0.6 | 35.0 | [Baidu Drive](https://pan.baidu.com/s/1LVcgdzX1TjtIQE_BnXB4RA), [Google Drive](https://drive.google.com/file/d/14l4bTXYsaE-NaMpmBa-dXmt7l3_TJogQ/view?usp=share_link)
SeaFormer-T_512x512_4x8_160k | 1.7 | 0.6 | 36.5 | [Baidu Drive](https://pan.baidu.com/s/1jCV8scTv--DRIlB0ml3yKg), [Google Drive](https://drive.google.com/file/d/1eIBkr2x5jv4eNaNUQpC-91tbIlktXSPi/view?usp=share_link)
SeaFormer-S_512x512_2x8_160k | 4.0 | 1.1 | 38.9 | [Baidu Drive](https://pan.baidu.com/s/1G0ypLXLThIRN7vo7zPBTqA), [Google Drive](https://drive.google.com/file/d/1eVLFdORpvdLS68hTJCN_SzGuHmepAFxT/view?usp=share_link)
SeaFormer-S_512x512_4x8_160k | 4.0 | 1.1 | 39.5 | [Baidu Drive](https://pan.baidu.com/s/1j7srQjz3F9WoGsIIjkSgBw), [Google Drive](https://drive.google.com/file/d/1hGXFVc7F-vLAKe3BLjqnS06_8Fo7CO-L/view?usp=share_link)
SeaFormer-B_512x512_2x8_160k | 8.6 | 1.8 | 41.2 | [Baidu Drive](https://pan.baidu.com/s/1CpA4-dWbENm1FSoRppaNwA), [Google Drive](https://drive.google.com/file/d/1H-GLdNzEViB2-QAtLdXpQVngUmMb_Vsa/view?usp=share_link)
SeaFormer-B_512x512_4x8_160k | 8.6 | 1.8 | 41.9 | [Baidu Drive](https://pan.baidu.com/s/1QEsoxlDz-EdAnVQn5vJJww), [Google Drive](https://drive.google.com/file/d/1flVg9imJTbgjcJrJiIn_3_lmYpaSKZuV/view?usp=share_link)
SeaFormer-L_512x512_2x8_160k | 14.0 | 6.5 | 43.0 | [Baidu Drive](https://pan.baidu.com/s/1gNPLfuJH21NZ55aQY3_6RQ), [Google Drive](https://drive.google.com/file/d/1AbbzfQIH6z7tJ8PGlnY1d0S1eEXkva8S/view?usp=share_link)
SeaFormer-L_512x512_4x8_160k | 14.0 | 6.5 | 43.8 | [Baidu Drive](https://pan.baidu.com/s/1Hybn3hKoxPdzRirVmqgjyw), [Google Drive](https://drive.google.com/file/d/1SUISoIpZujAYrxrvGPMJidzfYH2KYaAp/view?usp=share_link)

Results on Cityscapes
Model |  FLOPs | mIoU  | Link
--- |:---:|:---:|:---: |
SeaFormer-S_1024x512_1x8_160k  | 2.0 | 71.1 | [Baidu Drive](https://pan.baidu.com/s/1AbG61WTa_SsUwrU-nrZuYA), [Google Drive](https://drive.google.com/file/d/1MQ-nkCMyrzUF_SnWyNCrjra9bPTmr7Iq/view?usp=share_link)
SeaFormer-S_1024x1024_1x8_160k | 8.0 | 76.4 | [Baidu Drive](https://pan.baidu.com/s/1kJvndGxejy1x4Zc3w2DEkw), [Google Drive](https://drive.google.com/file/d/1jaia2FsZrxVkXzVY-BmIwQXr523feAxC/view?usp=share_link)
SeaFormer-B_1024x512_1x8_160k  | 3.4 | 72.2 | [Baidu Drive](https://pan.baidu.com/s/1V0iRz1KWssGiU_8Ai3bUzw), [Google Drive](https://drive.google.com/file/d/1xIcxkwjIJgvPrm4UK7UXH5A5ZkNrCqRt/view?usp=share_link)
SeaFormer-B_1024x1024_1x8_160k | 13.7| 77.7 | [Baidu Drive](https://pan.baidu.com/s/1RvaoX2UIynevbVn25aQT8A), [Google Drive](https://drive.google.com/file/d/1UgaSmQY1ZdFRokhHCB_u8HrDoNSodzIe/view?usp=share_link)

- The password of Baidu Drive is seaf

## Usage
Please see [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md) for dataset prepare.

clone the repository locally:
```
git clone https://github.com/fudan-zvg/SeaFormer
cd SeaFormer/seaformer-seg
mkdir -p modelzoos/classification
```
For training on SeaFormer_Tiny, run:
```
cp /cls_outdir/last.pth.tar modelzoos/classification/SeaFormer_T.pth
sh tools/dist_train.sh local_configs/seaformer/<config-file> <num-of-gpus-to-use> --work-dir /path/to/save/checkpoint
```
We use 8 gpus by default. If you use fewer gpus, you will need to increase the batch size to ensure that the total batch size remains the same.

To evaluate, run:
```
sh tools/dist_test.sh local_configs/seaformer/<config-file> <checkpoint-path> <num-of-gpus-to-use>
```

