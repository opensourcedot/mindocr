# Unified call interface
<a name="1"></a>
## 1. Module introduction
For MindOCR online inference, it is currently called mainly through command line scripts
```
python tools/infer/text/predict_system.py --image_dir {path_to_img or dir_to_imgs}  --det_algorithm DB++  --rec_algorithm CRNN
```
This call method usually needs to enter the warehouse directory and invoke the corresponding script to run. In order to make online reasoning more convenient, this module provides a unified call interface module, users can perform convenient and efficient online reasoning based on this module.

## 2. Online inference interface

This module currently provides an online inference interface `DetModel` for text detection tasks, and an online inference interface `RecModel` for text recognition tasks. Based on this interface, users can directly call the default configuration model, or adjust the relevant parameters for online inference. In addition, the interface also supports calling the configuration file for online inference, the configuration file format is detailed in the `yaml_configuration` document.

### 2.1 Text detection

The `DetModel` module is the `ModelEngine` unified call module for text detection models, which is used by default

```python
casemodel = DetModel(algo="DB++")
result = casemodel.infer("path_to_img")
```

Users can pass in a specified weight file and specify Settings based on this interface
```python
casemodel = DetModel(algo="DB++", model_dir="path_to_ckpt", mode=1)
result = casemodel.infer("path_to_img")
```

**Note:**
- mode (int): `0` indicates the static mode and `1` indicates the dynamic graph mode.
- algo (str) : indicates the configuration of the text detection algorithm.
- amp_level (str) : indicates the automatic mixing precision level. This setting applies only to `GPU` and `Ascend`.
- model_dir (str) : contains the directory that detects the model checkpoint `best-ckpt`, or the path to a specific checkpoint file.
- det_limit_side_len (int) : edge length limit for image resizing.
- det_limit_type (str) : indicates the limit type of image size. If minimum, the image will be resized by limiting the minimum side length to `limit_side_len`. If max, the image will be resized to `limit_side_len` by limiting the maximum value
- det_box_type (tuple) : indicates the type of the box represented by the text area.
- draw_img_save_dir (str) : A directory that holds visualization and detection/recognition/system prediction results.
- save_crop_res (bool) : Indicates whether to save the clipped image from the text detection result.
- crop_res_save_dir (str) : The directory where the cropped image is saved for the text box.
- visualize_out (bool) : Indicates whether to visualize the result and save the visualized image.

### 2.2 Character recognition

`RecModel` module is a unified call module of `ModelEngine` for text recognition models, and the default configuration of this interface is to use

```python
casemodel = RecModel(algo="CRNN")
res = casemodel.infer(["path_to_img"])
```

Users can pass in a specified weight file and specify Settings based on this interface
```python
casemodel = RecModel(algo="CRNN", model_dir="path_to_ckpt", mode=1)
result = casemodel.infer("path_to_img")
```

**Note:**
- mode (int): `0` indicates the static graph mode, and `1` indicates the dynamic graph mode.
- algo (str) : sets the text recognition algorithm.
- amp_level (str) : Automatic mixing precision level, this setting applies only to `GPU` and `Ascend`.
- model_dir (str) : contains the directory that identifies the model checkpoint `best-ckpt`, or the path to a specific checkpoint file.
- rec_image_shape (str) : C, H, W indicates the target image shape.
- max_wh_ratio: W/H is used to control the maximum width of the aspect ratio after the adjustment. Set W to longer text.
- rec_batch_mode (bool) : Whether to run recognition inference in batch mode, which is faster but may reduce accuracy due to filling or resizing to the same shape.
- Rec_char_dict_path (str) : indicates the path of the character dictionary. If not, it will be based on `rec_algorithm` and `red_model_dir`.
- draw_img_save_dir (str) : The directory where visualization and detection/recognition/system prediction results are saved.
- visualize_output (bool) : Indicates whether to visualize the result and save the visualized image.

### 3.3 Online inference based on profiles
The unified invocation interface module supports not only the default configuration for online inference, but also the online inference based on the configuration file. The current version of RecModel is supported, and we will support more model tasks in the future.

```python
config_file_path="crnn_resnet34.yaml"
casemodel = RecModel(init_with_config_file=True,config_file_path=config_file_path)
result = casemodel.infer(["path_to_image"])
```

**Note** : Configuration files need to be written in the MindOCR standard configuration file format, see the yaml_configuration document for details.

## Next plan
The current unified call interface has supported the character recognition model inference interface 'DetModel' and the character recognition model inference interface 'RecModel'. In the future, we will continue to support more types of model online inference interfaces, and launch a unified training and evaluation interface.
