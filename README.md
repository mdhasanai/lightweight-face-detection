# Face-Detector-1MB-with-landmark
## Features
 - Retinaface-mobile0.25 model converted into ncnn python/ opencv onnx/ pytorch python
 - Face-Detector-1MB slim 
 - 5 key points of face detection
 - Support onnx export
 - Network parameter and flop calculation


# Ultra-lightweight face detector with keypoint detection

Provides a series of face detectors suitable for mobile deployment including key face detectors: Modified the anchor size of [Retinaface-mobile0.25](https://github.com/biubug6/Pytorch_Retinaface) to make it more suitable for edge computing; Reimplemented [Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) and added key point detection and ncnn C++ The deployment function, in most cases, the accuracy is better than the original version.


## Requirments
- Ubuntu18.04
- Python3.7
- Pytorch1.2
- CUDA10.0 + CUDNN7.5

## accuracy
### Widerface test

 - Accuracy in wider face val (single-scale input resolution: **320*240**）
 
 method|Easy|Medium|Hard
------|--------|----------|--------
libfacedetection v1（caffe）|0.65 |0.5       |0.233
libfacedetection v2（caffe）|0.714 |0.585       |0.306
version-slim(original)|0.765     |0.662       |0.385
version-RFB(original)|0.784     |0.688       |**0.418**
version-slim(our)|0.795     |0.683       |0.34.5
version-RFB(our)|**0.814**     |**0.710**       |0.363
Retinaface-Mobilenet-0.25(our)  |0.811|0.697|0.376

- Accuracy in wider face val (single-scale input resolution:：**640*480**） 

method|Easy|Medium|Hard 
------|--------|----------|--------
libfacedetection v1（caffe）|0.741 |0.683       |0.421
libfacedetection v2（caffe）|0.773 |0.718       |0.485
version-slim(original)|0.757     |0.721       |0.511
version-RFB(original)|0.851     |0.81       |0.541
version-slim(our)|0.850     |0.808       |0.595
version-RFB(our)|0.865    |0.828       |0.622
Retinaface-Mobilenet-0.25(our)  |**0.873**|**0.836**|**0.638**

ps:  When testing, the long side is 320 or 640, and the image is scaled in equal proportions.

## Parameter and flop

method|parameter(M)|flop(M) 
------|--------|----------
version-slim(our)|0.343     |98.793
version-RFB(our)|0.359    |118.435
Retinaface-Mobilenet-0.25(our)  |0.426|193.921

ps: 320*240 as input


# Python inference
coming soon




## References
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
- [Retinaface (pytorch)](https://github.com/biubug6/Pytorch_Retinaface)
- [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}

