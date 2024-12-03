import os

import numpy as np
import mindspore as ms

from .modelengine import ModelEngine
from .utils import *

__all__ = ["DetModel"]
algo_to_model_name = {
    "DB": "dbnet_resnet50",
    "DB++": "dbnetpp_resnet50",
    "DB_MV3": "dbnet_mobilenetv3",
    "DB_PPOCRv3": "dbnet_ppocrv3",
    "PSE": "psenet_resnet152",
}


class DetModel(ModelEngine):
    def __init__(
        self,
        mode: int = 0,
        algo: str = "DB++",
        amp_level: str = "O0",
        model_dir: str = None,
        det_limit_side_len: int = 960,
        det_limit_type: str = "max",
        det_box_type: str = "quad",
        draw_img_save_dir: str = "./inference_results",
        save_crop_res: bool = False,
        crop_res_save_dir: str = "./output",
        visualize_output: bool = False,
        warmup: bool = False,
    ):
        """
        Initialize the DetModel with various parameters.
        Parameters:
            mode (int): 0 for graph mode, 1 for pynative mode.
            algo (str): detection algorithm.
            amp_level (str): Auto Mixed Precision level. This setting only works on GPU and Ascend.
            model_dir (str): directory containing the detection model checkpoint best.ckpt,
                or path to a specific checkpoint file.
            det_limit_side_len (int): side length limitation for image resizing.
            det_limit_type (str):   limitation type for image resize. If min, images
                will be resized by limiting the minimum side length to `limit_side_len`
                (prior to accuracy). If max, images will be resized by limiting the maximum
                side length to `limit_side_len` (prior to speed). Default: max.
            det_box_type (tuple): box type for text region representation.
            draw_img_save_dir (str): Dir to save visualization and detection/recogintion/system prediction results.
            save_crop_res (bool): Whether to save images cropped from text detection results.
            crop_res_save_dir (str): Dir to save the cropped images for text boxes.
            visualize_out (bool): Whether to visualize results and save the visualized image.
            warmup (bool): None
        """
        model_name = algo_to_model_name[algo]
        super().__init__(
            mode=mode,
            task="det",
            algo=algo,
            amp_level=amp_level,
            model_dir=model_dir,
            model_name=model_name,
            det_limit_side_len=det_limit_side_len,
            det_limit_type=det_limit_type,
            det_box_type=det_box_type,
        )
        self.mode = mode
        self.algo = algo
        self.amp_level = amp_level
        self.model_dir = model_dir
        self.det_limit_side_len = det_limit_side_len
        self.det_limit_type = det_limit_type
        self.det_box_type = det_box_type
        self.draw_img_save_dir = draw_img_save_dir
        self.save_crop_res = save_crop_res
        self.crop_res_save_dir = crop_res_save_dir
        self.visualize_output = visualize_output
        self.model_name = model_name
        self.warmup = warmup

        os.makedirs(self.draw_img_save_dir, exist_ok=True)


    def infer(self, img_or_path, **kwargs):
        self.model.set_train(mode=False)
        data = self.preprocess(img_or_path)
        input_np = data["image"]
        if len(input_np.shape) == 3:
            net_input = np.expand_dims(input_np, axis=0)
        net_output = self.model(ms.Tensor(net_input))
        det_res = self.postprocess(net_output, data)
        det_res_final = validate_det_res(det_res, data["image_ori"].shape[:2], min_poly_points=3, min_area=3)
        return det_res_final
