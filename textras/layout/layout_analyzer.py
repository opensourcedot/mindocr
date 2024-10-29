import os
import sys
import cv2
import logging
from typing import Dict, List, Tuple

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

from mindocr import build_model

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../../")))

from deploy.py_infer.src.data_process.preprocess.transforms.layout_transforms import *
from mindocr.postprocess.layout_postprocess import YOLOv8Postprocess
from mindocr.models.backbones.mindcv_models.utils import auto_map, download_pretrained

logger = logging.getLogger("Textras")

__all__ = ["LayoutAnalyzer", "visualize"]

yolov8_model_config = {
    "backbone": {
        "name": "yolov8_backbone",
        "depth_multiple": 0.33,
        "width_multiple": 0.25,
        "max_channels": 1024,
        "nc": 5,
        "stride": [8, 16, 32, 64],
        "reg_max": 16,
        "sync_bn": False,
        "out_channels": [64, 128, 192, 256],
    },
    "neck": {
        "name": "YOLOv8Neck",
        "index": [20, 23, 26, 29],
    },
    "head": {
        "name": "YOLOv8Head",
        "nc": 5,
        "reg_max": 16,
        "stride": [8, 16, 32, 64],
        "sync_bn": False,
    },
}

yolov8_ckpt_url = "https://download.mindspore.cn/toolkits/mindocr/yolov8/yolov8n-4b9e8004.ckpt"

class LayoutAnalyzer(object):
    '''
    Infer model for layout analysis

    Attributes:
        model (Model): model for layout analysis
    
    Methods:
        __init__(self, model_name, ckpt_load_path, amp_level): Initialize LayoutAnalyzer
    '''
    def __init__(self, args, amp_level: str = "O0"):
        '''
        Initialize LayoutAnalyzer

        Args:
            model_name (str): model name
            ckpt_load_path (str): checkpoint load path
            amp_level (str): amp level
        '''
        if ms.get_context("device_target") == "GPU" and amp_level == "O3":
            logger.warning(
                "Detection model prediction does not support amp_level O3 on GPU currently. "
                "The program has switched to amp_level O2 automatically."
            )
            amp_level = "O2"
        if not hasattr(args, "layout_model_dir"):
            url_cfg = {"url": yolov8_ckpt_url}
            self.layout_ckpt_path = download_pretrained(url_cfg)
        else: 
            assert os.path.exists(ckpt_load_path) == True, f"ckpt_load_path {args.layout_model_dir} not exists"
            self.layout_ckpt_path = args.layout_model_dir
      
        self.model = build_model(yolov8_model_config, ckpt_load_path=self.layout_ckpt_path, amp_level=amp_level)
        self.model.set_train(False)

    def infer(self, image_path: str):
        """
            Args:
        img_or_path: str for img path or np.array for RGB image
        do_visualize: visualize preprocess and final result and save them

            Return:
        det_res_final (dict): detection result with keys:
                            - polys: np.array in shape [num_polygons, 4, 2] if det_box_type is 'quad'. Otherwise,
                              it is a list of np.array, each np.array is the polygon points.
                            - scores: np.array in shape [num_polygons], confidence of each detected text box.
        data (dict): input and preprocessed data with keys: (for visualization and debug)
            - image_ori (np.ndarray): original image in shape [h, w, c]
            - image (np.ndarray): preprocessed image feed for network, in shape [c, h, w]
            - shape (list): shape and scaling information [ori_h, ori_w, scale_ratio_h, scale_ratio_w]
        """
        ms.set_context(mode=1)
        image = self.load_image(image_path)
        image = self.preprocess(image)
        input_data = Tensor([image], ms.float32)
        preds = self.model(input_data)
        self.shape = input_data.shape
        return image, self.postprocess(preds)
        

    def load_image(self, image_path: str) -> np.ndarray:
        '''
        Load image

        Args:
            image_path (str): image path

        Returns:
            np.ndarray: image
        '''
        assert os.path.exists(image_path) == True, f"image_path {image_path} not exists"
        image = cv2.imread(image_path)
        assert image is not None, f"image_path {image_path} is not a valid image"
        return image
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # resize image
        h_ori, w_ori = image.shape[:2]  # orig hw
        hw_ori = np.array([h_ori, w_ori])
        r = 800 / max(h_ori, w_ori)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            image = cv2.resize(image, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
        letterbox = letterbox(scaleup=False)
        image_norm = image_norm(scale=255.0)
        image_transpose = image_transpose(bgr2rgb=True, hwc2chw=True)
        raw_img_data = {"image": image, "raw_img_shape": hw_ori, "target_size": 800}
        image_data = letterbox(raw_img_data)
        image_data = image_norm(image_data)
        image_data = image_transpose(image_data)
        image = np.ascontiguousarray(image_data["image"])
        return image
    
    def postprocess(self, preds: np.ndarray) -> Dict:
        # postprocess
        postprocess_ops = YOLOv8Postprocess(conf_thres=0.01, iou_thres=0.7, conf_free=True)
        results = postprocess_ops(preds, [self.shape], meta_info=([0], [self.hw_ori], [self.hw_scale], [self.hw_pad]))
        return results

def visualize(image_path, results, conf_thres=0.8, save_path: str =  ""):
    from PIL import Image
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches

    img = Image.open(image_path)
    img_cv = cv2.imread(image_path)
    
    fig, ax = plt.subplots()
    ax.imshow(img)

    category_dict = {1: 'text', 2: 'title', 3: 'list', 4: 'table', 5: 'figure'}
    color_dict = {1: (255, 0, 0), 2: (0, 0, 255), 3: (0, 255, 0), 4: (0, 255, 255), 5: (255, 0, 255)}

    for item in results:
        category_id = item['category_id']
        bbox = item['bbox']
        score = item['score']

        if score < conf_thres:
            continue
        
        left, bottom, w, h = bbox
        right = left + w
        top = bottom + h

        cv2.rectangle(img_cv, (int(left), int(bottom)), (int(right), int(top)), color_dict[category_id], 2)

        label = '{} {:.2f}'.format(category_dict[category_id], score)
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv2.rectangle(img_cv, (int(left), int(bottom - label_size[1] - base_line)), (int(left + label_size[0]), int(bottom)), color_dict[category_id], cv2.FILLED)
        cv2.putText(img_cv, label, (int(left), int(bottom - base_line)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if save_path:
        cv2.imwrite(save_path, img_cv)
    else:
        plt.axis('off')
        plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        plt.show()
