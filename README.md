## General
- Implementation crowd-counting to traffic domain and counting vehicle on road
- paper reference: [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/pdf/1802.10062.pdf)
## Requirements
- torch
- torchvision
- pillow
- opencv-python
- numpy
- GPU-CUDA (optional)
## Training
```
python3 train.py --train <path to folder training set> -cfg model.json --use_pretrain --density <path to dataset density> --cuda -i 200 -lr 0.000001 -wk 3 -bs 4
```
`--cuda`: flag to use GPU NVIDIA to training phase. If you don't want use GPUs just skip it.

`-wk`:(worker) multi-processor to train, it make training phase more faster( if you got strong cpu like i5, ryzen5, xeon, ...)

`-bs`: batch-size

`-lr`: learning rate, make it small enough to training, can you monitor with tensorboard and watch `total gradient graph` it must be fluctuate
## Load from checkpoint and continue to train:
```
python3 train.py --train <path to folder training set> -cfg model.json --density <path to dataset density> --cuda -i 200 -lr 0.000001 -wk 3 -bs 4 --checkpoint
```
## Validation
```
python3 eval.py --test <path to folder training set> -cfg model.json --density <path to eval density> --cuda -wk 3 -bs 4
```
## Test with video
```
python3 video_test.py --input <path to video> -cfg model.json --cuda -l ./logs
```