import logging
import os

import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore.common import dtype as mstype

from .utils import *
from .modelengine import ModelEngine
from ..utils.visualize import show_imgs

__all__=["RecModel"]
algo_to_model_name = {
    "CRNN": "crnn_resnet34",
    "RARE": "rare_resnet34",
    "CRNN_CH": "crnn_resnet34_ch",
    "RARE_CH": "rare_resnet34_ch",
    "SVTR": "svtr_tiny",
    "SVTR_PPOCRv3_CH": "svtr_ppocrv3_ch",
}
logger = logging.getLogger("mindocr")


class RecModel(ModelEngine):
    def __init__(
        self,
        mode: int = 0,
        algo: str = "CRNN",
        amp_level: str = "O0",
        model_dir: str = None,
        rec_image_shape: str = "3,32,320",
        rec_batch_mode: bool = True,
        rec_batch_num: int = 8,
        max_text_length: int = 25,
        rec_char_dict_path: str = None,
        use_space_char: bool = True,
        vis_font_path: str = "docs/fonts/simfang.ttf",
        drop_score: float = 0.5,
        draw_img_save_dir: str = "./inference_results",
        visualize_output: bool = False,
        warmup: bool = False,
        init_with_config_file: bool = False,
        config_file_path: str = None
    ):
        """
        Initialize the DetModel with various parameters.  
        Parameters:  
            mode (int): 0 for graph mode, 1 for pynative mode.  
            algo (str): detection algorithm. 
            amp_level (str): Auto Mixed Precision level. This setting only works on GPU and Ascend.  
            model_dir (str): directory containing the recognition model checkpoint best.ckpt, 
                or path to a specific checkpoint file.  
            rec_image_shape (str):C, H, W for target image shape. max_wh_ratio=W/H will be used to 
                control the maximum width after aspect-ratio-kept resizing. Set W larger for longer text.
            rec_batch_mode (bool): Whether to run recognition inference in batch-mode, which is 
                faster but may degrade the accuracy due to padding or resizing to the same shape.
            rec_batch_num (int): None
            max_text_length (int): None
            rec_char_dict_path (str): path to character dictionary. If None, will pick according 
                to rec_algorithm and red_model_dir.
            use_space_char (bool): None
            vis_font_path (str): None
            drop_score (float): None
            draw_img_save_dir (str): Dir to save visualization and detection/recogintion/system prediction results.  
            visualize_output (bool): Whether to visualize results and save the visualized image. 
            warmup (bool): None
        """
        if init_with_config_file:
            super().__init__(
                init_with_config_file = init_with_config_file,
                config_file_path = config_file_path
            )
        else:
            model_name = algo_to_model_name[algo]
            super().__init__(
                mode = mode,
                task = "rec",
                algo = algo,
                amp_level = amp_level,
                model_dir = model_dir,
                model_name = model_name,
                rec_batch_mode = rec_batch_mode
            )
            self.mode = mode
            self.algo = algo
            self.amp_level = amp_level
            self.model_dir = model_dir
            self.rec_image_shape = rec_image_shape
            self.rec_batch_mode = rec_batch_mode
            self.rec_batch_num = rec_batch_num
            self.max_text_length = max_text_length
            self.rec_char_dict_path = rec_char_dict_path
            self.use_space_char = use_space_char
            self.vis_font_path = vis_font_path
            self.drop_score = drop_score
            self.draw_img_save_dir = draw_img_save_dir
            self.visualize_output = visualize_output
            self.warmup = warmup
            self.model_name = model_name

        self.rec_batch_mode = rec_batch_mode
        self.rec_batch_num = rec_batch_num

        self.cast_pred_fp32 = amp_level != "O0"
        if self.cast_pred_fp32:
            self.cast = ops.Cast()
        
        self.vis_dir = draw_img_save_dir
        os.makedirs(self.vis_dir, exist_ok=True)

    def infer(self, img_or_path_list, do_visualize=False):
        self.model.set_train(False)
        assert isinstance(img_or_path_list, list), "Input for text recognition must be list of images or image paths."
        if self.rec_batch_mode:
            rec_res_all_crops = self.run_batchwise(img_or_path_list, do_visualize)
        else:
            rec_res_all_crops = []
            for i, img_or_path in enumerate(img_or_path_list):
                rec_res = self.run_single(img_or_path, i, do_visualize)
                rec_res_all_crops.append(rec_res)

        return rec_res_all_crops

    def run_batchwise(self, img_or_path_list: list, do_visualize=False):
        """
        Run text recognition serially for input images

            Args:
            img_or_path_list: list of str for img path or np.array for RGB image
            do_visualize: visualize preprocess and final result and save them

            Return:
            rec_res: list of tuple, where each tuple is  (text, score) - text recognition result for each input image
                in order.
                    where text is the predicted text string, score is its confidence score.
                    e.g. [('apple', 0.9), ('bike', 1.0)]
        """
        rec_res = []
        num_imgs = len(img_or_path_list)

        for idx in range(0, num_imgs, self.rec_batch_num):  # batch begin index i
            batch_begin = idx
            batch_end = min(idx + self.rec_batch_num, num_imgs)
            logger.info(f"Rec img idx range: [{batch_begin}, {batch_end})")

            # preprocess
            img_batch = []
            for j in range(batch_begin, batch_end):  # image index j
                data = self.preprocess(img_or_path_list[j])
                img_batch.append(data["image"])
                if do_visualize:
                    fn = os.path.basename(data.get("img_path", f"crop_{j}.png")).rsplit(".", 1)[0]
                    show_imgs(
                        [data["image"]],
                        title=fn + "_rec_preprocessed",
                        mean_rgb=[127.0, 127.0, 127.0],
                        std_rgb=[127.0, 127.0, 127.0],
                        is_chw=True,
                        show=False,
                        save_path=os.path.join(self.vis_dir, fn + "_rec_preproc.png"),
                    )

            img_batch = np.stack(img_batch) if len(img_batch) > 1 else np.expand_dims(img_batch[0], axis=0)

            # infer
            net_pred = self.model(ms.Tensor(img_batch))
            if self.cast_pred_fp32:
                if isinstance(net_pred, list) or isinstance(net_pred, tuple):
                    net_pred = [self.cast(p, mstype.float32) for p in net_pred]
                else:
                    net_pred = self.cast(net_pred, mstype.float32)

            # postprocess
            batch_res = self.postprocess(net_pred)
            rec_res.extend(list(zip(batch_res["texts"], batch_res["confs"])))

        return rec_res

    def run_single(self, img_or_path, crop_idx=0, do_visualize=True):
        """
        Text recognition inference on a single image
        Args:
            img_or_path: str for image path or np.array for image rgb value

        Return:
            dict with keys:
                - texts (str): preditive text string
                - confs (int): confidence of the prediction
        """
        # preprocess
        data = self.preprocess(img_or_path)

        # visualize preprocess result
        if do_visualize:
            fn = os.path.basename(data.get("img_path", f"crop_{crop_idx}.png")).rsplit(".", 1)[0]
            show_imgs(
                [data["image"]],
                title=fn + "_rec_preprocessed",
                mean_rgb=[127.0, 127.0, 127.0],
                std_rgb=[127.0, 127.0, 127.0],
                is_chw=True,
                show=False,
                save_path=os.path.join(self.vis_dir, fn + "_rec_preproc.png"),
            )
        logger.info(f"Origin image shape: {data['image_ori'].shape}")
        logger.info(f"Preprocessed image shape: {data['image'].shape}")

        # infer
        input_np = data["image"]
        if len(input_np.shape) == 3:
            net_input = np.expand_dims(input_np, axis=0)

        net_pred = self.model(ms.Tensor(net_input))
        if self.cast_pred_fp32:
            if isinstance(net_pred, list) or isinstance(net_pred, tuple):
                net_pred = [self.cast(p, mstype.float32) for p in net_pred]
            else:
                net_pred = self.cast(net_pred, mstype.float32)

        # postprocess
        rec_res = self.postprocess(net_pred)

        rec_res = (rec_res["texts"][0], rec_res["confs"][0])

        logger.info(f"Crop {crop_idx} rec result: {rec_res}")

        return rec_res
