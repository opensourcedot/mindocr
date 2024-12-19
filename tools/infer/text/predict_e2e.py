import json
import logging
import os

import cv2
import numpy as np
from config import parse_args
from postprocess import Postprocessor
from preprocess import Preprocessor
from utils import draw_e2e_res, get_ckpt_file, get_image_paths

import mindspore as ms
from mindspore import ops

from mindocr import build_model
from mindocr.utils.logger import set_logger

# map algorithm name to model name (which can be checked by `mindocr.list_models()`)
# NOTE: Modify it to add new model for inference.
algo_to_model_name = {
    "PG": "pgnet_resnet50",
}
logger = logging.getLogger("mindocr")


class TextEnd2End(object):
    def __init__(self, args):
        ckpt_dir = args.e2e_model_dir
        if ckpt_dir is None:
            pretrained = True
            ckpt_load_path = None
        else:
            ckpt_load_path = get_ckpt_file(ckpt_dir)
            pretrained = False
        assert args.e2e_algorithm in algo_to_model_name, (
            f"Invalid e2e_algorithm {args.e2e_algorithm}. "
            f"Supported algorithms are {list(algo_to_model_name.keys())}"
        )

        model_name = algo_to_model_name[args.e2e_algorithm]
        amp_level = args.e2e_amp_level
        self.model = build_model(model_name, pretrained=pretrained, ckpt_load_path=ckpt_load_path, amp_level=amp_level)
        self.model.set_train(False)

        self.cast_pred_fp32 = amp_level != "O0"
        if self.cast_pred_fp32:
            self.cast = ops.Cast()
        logger.info(
            "Init model: {} --> {}. Model weights loaded from {}".format(
                args.e2e_algorithm, model_name, "pretrained url" if pretrained else ckpt_load_path
            )
        )

        # build preprocess and postprocess
        self.preprocess = Preprocessor(
            task="e2e",
            algo=args.e2e_algorithm,
            e2e_limit_side_len=args.e2e_limit_side_len,
            e2e_limit_type=args.e2e_limit_type,
        )

        self.postprocess = Postprocessor(task="e2e", algo=args.e2e_algorithm)

        self.vis_dir = args.draw_img_save_dir
        os.makedirs(self.vis_dir, exist_ok=True)

        self.box_type = "poly"
        self.visualize_preprocess = True

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img_or_path):
        # preprocess
        ori_im = cv2.imread(img_or_path)
        data = self.preprocess(img_or_path)
        # infer
        input_np = data["image"]
        if len(input_np.shape) == 3:
            net_input = np.expand_dims(input_np, axis=0)

        net_output = self.model(ms.Tensor(net_input))

        # postprocess
        points, strs = self.postprocess(net_output, data)
        dt_boxes = self.filter_tag_det_res_only_clip(points, ori_im.shape)
        return dt_boxes, strs


def save_res(boxes_all, text_scores_all, img_path, save_path):
    lines = []
    img_name = os.path.basename(img_path).rsplit(".", 1)[0]
    save_path = os.path.join(save_path, img_name + "_e2e_rec_result.txt")
    boxes = boxes_all
    text_scores = text_scores_all

    res = []  # result for current image
    for j in range(len(boxes)):
        res.append(
            {
                "transcription": text_scores[j],
                "points": np.array(boxes[j]).astype(np.int32).tolist(),
            }
        )

    img_res_str = os.path.basename(img_path) + "\t" + json.dumps(res, ensure_ascii=False) + "\n"
    lines.append(img_res_str)

    with open(save_path, "w") as f:
        f.writelines(lines)
        f.close()


def predict_rec_e2e(image_path):
    # parse args
    args = parse_args()
    set_logger(name="mindocr")
    save_dir = "./inference_results"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    args.draw_img_save_dir = save_dir
    args.image_dir = image_path
    save_folder = args.draw_img_save_dir

    save_folder, _ = os.path.splitext(save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    img_paths = get_image_paths(args.image_dir)

    ms.set_context(mode=args.mode)

    # init detector
    text_detect = TextEnd2End(args)

    boxes_all, text_scores_all = [], []
    for i, img_path in enumerate(img_paths):
        logger.info(f"\nInfering [{i+1}/{len(img_paths)}]: {img_path}")
        points, strs = text_detect(img_path)
        boxes_all.append(points)
        text_scores_all.append(strs)
        save_res(boxes_all, text_scores_all, img_path, save_folder)
        src_im = draw_e2e_res(points, strs, img_path)
        img_res_path = os.path.join(save_dir, "text_e2e_ocr_res.png")
        cv2.imwrite(img_res_path, src_im)


if __name__ == "__main__":
    args = parse_args()
    image_path = args.image_dir
    predict_rec_e2e(image_path)
