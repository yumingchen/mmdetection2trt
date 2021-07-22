### mmdetection2trt
#### 本项目不依赖torch，仅用numpy实现mmdetection model到tensorrt的部署。
1. [mmdetection-to-tensorrt](https://github.com/grimoire/mmdetection-to-tensorrt) 实现了mmdetection多个模型到tensorrt的转换.
2. 但是mmdetection-to-tensorrt提供的tensorrt推理示例是依赖torch的。这样不利于nano、tx2等设备部署。
#### 本项目基于[mmdetection-to-tensorrt](https://github.com/grimoire/mmdetection-to-tensorrt) 用pure numpy实现了mmdetection模型到tensorrt模型的推理。
### Requirement
1. 安装[mmdetection](https://github.com/open-mmlab/mmdetection). 用于训练自己的模型。
2. 安装[mmdetection-to-tensorrt](https://github.com/grimoire/mmdetection-to-tensorrt). 用于转换模型。

### Usage
#### 安装完上述环境后，可用下列方式实现mmdetection模型到tensorrt的转换，以及tensorrt模型的inference。
#### 这里用自己训练cascade rcnn的为例：
#### clone 
```bash
git clone https://github.com/yumingchen/mmdetection2trt.git
```
#### mmdetection to tensorrt
- step 1： 根据自己的情况修改mmdet_to_trt.py中文件路径
```python
# save_path保存用于mmdetection-to-tensorrt推理的模型，对于本项目，可以不用。
save_path = '.../cascade_rcnn/epoch_37_trt.pth'
# checkpoint：自己训练的模型参数路径，
checkpoint = '.../cascade_rcnn/epoch_37.pth'
# cfg_path：训练时的参数文件，训练时会在保存模型的路径下生产完成的配置文件。
cfg_path = '.../cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco_comp2.py'
```
- step 2：
```bash
python mmdet_to_trt.py
# or CLI
mmdet2trt ${CONFIG_PATH} ${CHECKPOINT_PATH} ${OUTPUT_PATH}
# eg：mmdet2trt .../cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco_comp2.py .../cascade_rcnn/epoch_37.pth .../cascade_rcnn/epoch_37_pth.engine
```
- step 3：执行完， 会生成tensorrt模型：engine_file = '../cascade_rcnn/epoch_37_pth.engine'
#### tensorrt inference
- step 1：修改inference_trt.py中的相关路径
```python
# 安装mmdetection-to-tensorrt的过程会安装amirstan_plugin
PLUGIN_LIBRARY = "/home/cym/programfiles/amirstan_plugin/build/lib/libamirstan_plugin.so"
ctypes.CDLL(PLUGIN_LIBRARY)
# tensorrt模型文件：python mmdet_to_trt.py生成。
engine_path = '.../cascade_rcnn/epoch_37_pth.engine'
```
- step 2: 推理
```bash 
python inference_trt.py
```