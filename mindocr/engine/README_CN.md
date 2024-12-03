# 统一调用接口
## 1. 模块简介
对于MindOCR在线推理，当前主要通过命令行脚本进行调用
```
python tools/infer/text/predict_system.py --image_dir {path_to_img or dir_to_imgs}  --det_algorithm DB++  --rec_algorithm CRNN
```
该调用方式通常需要进入到仓库目录下，调用相应脚本运行。为便于更加方便进行在线推理，本模块提供了统一调用接口模块ModelEngine，用户可基于本模块进行方便而高效的在线推理。

## 2. 在线推理接口
本模块当前提供了文字检测任务在线推理接口`DetModel`，以及文字识别任务在线推理接口`RecModel`。用户可基于该接口直接调用默认配置模型，也可调整相关的参数进行在线推理。此外，该接口也支持调用配置文件进行在线推理，配置文件格式详细参见`yaml_configuration`文档。

### 2.1 文字检测
DetModel模块是ModelEngine关于文字检测模型的统一调用模块，该接口默认配置下使用方式为

```python
casemodel = DetModel(algo="DB++")
result = casemodel.infer("path_to_img")
```

用户可基于该接口传入指定权重文件以及指定设置
```python
casemodel = DetModel(algo="DB++", model_dir="path_to_ckpt", mode=1)
result = casemodel.infer("path_to_img")
```

**注意：**
- mode (int): `0`表示静态模式，`1`表示动态图模式。
- algo（str）：文字检测算法配置。
- amp_level (str)：自动混合精度级别。此设置仅适用于`GPU`和`Ascend`。
- model_dir (str)：包含检测模型检查点`best.ckpt`的目录，或特定检查点文件的路径。
- det_limit_side_len (int)：图像调整大小的边长度限制。
- det_limit_type (str)：图像大小的限制类型。如果最小，图像将通过将最小边长限制为`limit_side_len`来调整大小。如果是max，图像将通过限制最大值来调整大小边长为`limit_side_len`
- det_box_type (tuple)：文本区域表示的框类型。
- draw_img_save_dir (str)：保存可视化和检测/识别/系统预测结果的目录。
- save_crop_res (bool)：是否保存从文本检测结果中裁剪的图像。
- crop_res_save_dir (str)：为文本框保存裁剪图像的目录。
- visualize_out (bool)：是否可视化结果并保存可视化图像。

### 2.2 文字识别
RecModel模块是ModelEngine关于文字识别模型的统一调用模块，该接口默认配置下使用方式为

```python
casemodel = RecModel(algo="CRNN")
res = casemodel.infer(["path_to_img"])
```

用户可基于该接口传入指定权重文件以及指定设置
```python
casemodel = RecModel(algo="CRNN", model_dir="path_to_ckpt", mode=1)
result = casemodel.infer("path_to_img")
```

**注意：**
- mode (int): `0`表示静态图模式，`1`表示动态图模式。
- algo（str）：文字识别算法设置。
- amp_level (str)：自动混合精度级别，此设置仅适用于`GPU`和`Ascend`。
- model_dir (str)：包含识别模型检查点`best.ckpt`的目录，或特定检查点文件的路径。
- rec_image_shape (str)：C，H，W为目标图像形状.
- max_wh_ratio：W/H将用于控制长宽比保持调整大小后的最大宽度。将W设置为更长的文本。
- rec_batch_mode (bool)：是否以批处理模式运行识别推理，该模式下速度更快，但由于填充或调整大小到相同的形状，可能会降低准确性。
- Rec_char_dict_path (str)：字符字典的路径。如果没有将根据到`rec_algorithm`和`red_model_dir`。
- draw_img_save_dir (str)：保存可视化和检测/识别/系统预测结果的目录。
- visualize_output (bool)：是否将结果可视化并保存可视化图像。

### 3.3 基于配置文件在线推理

统一调用接口模块除支持调用默认配置进行在线推理外，还支持基于配置文件的在线推理方式，当前版本RecModel已支持，我们将在接下来支持更多模型任务。

```python
config_file_path="crnn_resnet34.yaml"
casemodel = RecModel(init_with_config_file=True,config_file_path=config_file_path)
result = casemodel.infer(["path_to_image"])
```

**注意**：配置文件需要按照MindOCR标准配置文件格式编写，详细参见`yaml_configuration`文档。

## 下一步计划
当前统一调用接口已支持文字识别模型推理接口`DetModel`以及文字识别模型推理接口`RecModel`，后续我们将继续支持更多类型模型在线推理接口，以及推出统一的训练、评估接口。
