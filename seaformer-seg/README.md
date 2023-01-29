# SeaFormer


## Requirements

- pytorch 1.5+
- mmcv-full==1.3.14


## Main results

Model | Params | FLOPs | mIoU(ss)    | Link
--- |:---:|:---:|:---:|:---: |
SeaFormer-T_512x512_2x8_160k | 1.7 | 0.6 | 35.0 | [Baidu Drive](), [Google Drive]()
SeaFormer-T_512x512_4x8_160k | 1.7 | 0.6 | 35.8 | [Baidu Drive](), [Google Drive]()
SeaFormer-S_512x512_2x8_160k | 4.0 | 1.1 | 38.1 | [Baidu Drive](), [Google Drive]()
SeaFormer-S_512x512_4x8_160k | 4.0 | 1.1 | 39.4 | [Baidu Drive](), [Google Drive]()
SeaFormer-B_512x512_2x8_160k | 8.6 | 1.8 | 40.2 | [Baidu Drive](), [Google Drive]()
SeaFormer-B_512x512_4x8_160k | 8.6 | 1.8 | 41.0 | [Baidu Drive](), [Google Drive]()
SeaFormer-L_512x512_2x8_160k | 14.0 | 6.5 | 42.7 | [Baidu Drive](), [Google Drive]()
SeaFormer-L_512x512_4x8_160k | 14.0 | 6.5 | 43.7 | [Baidu Drive](), [Google Drive]()

- The password of Baidu Drive is seaf

## Usage
Please see [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md) for dataset prepare.

For training, run:
```
sh tools/dist_train.sh local_configs/seaformer/<config-file> <num-of-gpus-to-use> --work-dir /path/to/save/checkpoint
```
To evaluate, run:
```
sh tools/dist_test.sh local_configs/seaformer/<config-file> <checkpoint-path> <num-of-gpus-to-use>
```


## Citation

If you find our work helpful to your experiments, please cite with:
```

```


