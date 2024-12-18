"""
Text recognition inference

Example:
    $ python tools/infer/text/predict_rec.py  --image_dir {path_to_img} --rec_algorithm CRNN
    $ python tools/infer/text/predict_rec.py  --image_dir {path_to_img} --rec_algorithm CRNN_CH
"""
import logging
import os
import sys
from time import time

import numpy as np

import mindspore as ms
import mindspore.ops as ops
from mindspore.common import dtype as mstype

from mindocr import build_model
from mindocr.utils.logger import set_logger
from mindocr.utils.visualize import show_imgs
from config import parse_args
from postprocess import Postprocessor
from preprocess import Preprocessor
from utils import get_ckpt_file, get_image_paths

# map algorithm name to model name (which can be checked by `mindocr.list_models()`)
# NOTE: Modify it to add new model for inference.
algo_to_model_name = {"CAN": "can"}
logger = logging.getLogger("mindocr")


class TextRecognizer(object):
    def __init__(self, args):
        self.batch_num = args.rec_batch_num
        self.batch_mode = args.rec_batch_mode
        logger.info(
            "recognize in {} mode {}".format(
                "batch" if self.batch_mode else "serial",
                "batch_size: " + str(self.batch_num) if self.batch_mode else "",
            )
        )

        # build model for algorithm with pretrained weights or local checkpoint
        ckpt_dir = args.rec_model_dir
        if ckpt_dir is None:
            pretrained = True
            ckpt_load_path = None
        else:
            ckpt_load_path = get_ckpt_file(ckpt_dir)
            pretrained = False
        assert args.rec_algorithm in algo_to_model_name, (
            f"Invalid rec_algorithm {args.rec_algorithm}. "
            f"Supported recognition algorithms are {list(algo_to_model_name.keys())}"
        )
        model_name = algo_to_model_name[args.rec_algorithm]

        amp_level = args.rec_amp_level
        if args.rec_algorithm.startswith("SVTR") and amp_level != "O2":
            logger.warning(
                "SVTR recognition model is optimized for amp_level O2. ampl_level for rec model is changed to O2"
            )
            amp_level = "O2"
        if ms.get_context("device_target") == "GPU" and amp_level == "O3":
            logger.warning(
                "Recognition model prediction does not support amp_level O3 on GPU currently. "
                "The program has switched to amp_level O2 automatically."
            )
            amp_level = "O2"
        self.model = build_model(model_name, pretrained=pretrained, ckpt_load_path=ckpt_load_path, amp_level=amp_level)

        self.model.set_train(False)
        self.cast_pred_fp32 = amp_level != "O0"
        if self.cast_pred_fp32:
            self.cast = ops.Cast()
        logger.info(
            "Init recognition model: {} --> {}. Model weights loaded from {}".format(
                args.rec_algorithm, model_name, "pretrained url" if pretrained else ckpt_load_path
            )
        )

        # build preprocess and postprocess
        # NOTE: most process hyper-params should be set optimally for the pick algo.
        self.preprocess = Preprocessor(
            task="rec",
            algo=args.rec_algorithm,
            rec_image_shape=args.rec_image_shape,
            rec_batch_mode=self.batch_mode,
            rec_batch_num=self.batch_num,
        )

        # TODO: try GeneratorDataset to wrap preprocess transform on batch for possible speed-up.
        #  if use_ms_dataset: ds = ms.dataset.GeneratorDataset(wrap_preprocess, ) in run_batchwise
        self.postprocess = Postprocessor(
            task="rec", algo=args.rec_algorithm, rec_char_dict_path=args.rec_char_dict_path
        )

        self.vis_dir = args.draw_img_save_dir
        os.makedirs(self.vis_dir, exist_ok=True)

    def __call__(self, img_or_path_list: list, do_visualize=False):
        """
        Run text recognition serially for input images

        Args:
            img_or_path_list: list of str for img path or np.array for RGB image
            do_visualize: visualize preprocess and final result and save them

        Return:
            list of dict, each contains the follow keys for recognition result.
            e.g. [{'texts': 'abc', 'confs': 0.9}, {'texts': 'cd', 'confs': 1.0}]
                - texts: text string
                - confs: prediction confidence
        """

        assert isinstance(img_or_path_list, list), "Input for text recognition must be list of images or image paths."
        logger.info(f"num images for rec: {len(img_or_path_list)}")
        rec_res_all_crops = self.run_batchwise(img_or_path_list, do_visualize)
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

        for idx in range(0, num_imgs, self.batch_num):  # batch begin index i
            batch_begin = idx
            batch_end = min(idx + self.batch_num, num_imgs)
            logger.info(f"Rec img idx range: [{batch_begin}, {batch_end})")

            # preprocess
            img_batch = []
            data = {}
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

            image_mask = ops.ones(img_batch.shape, ms.float32)
            label = ops.ones((1, 36), ms.int64)
            image = ms.Tensor(img_batch)
            net_pred = self.model(image, image_mask, label)

            if self.cast_pred_fp32:
                if isinstance(net_pred, list) or isinstance(net_pred, tuple):
                    net_pred = [self.cast(p, mstype.float32) for p in net_pred]
                else:
                    net_pred = self.cast(net_pred, mstype.float32)

            # postprocess
            batch_res = self.postprocess(net_pred)
            rec_res = batch_res["texts"]

        return rec_res


def save_rec_res(rec_res_all, save_path="./rec_results.txt"):
    with open(save_path, "w") as file:
        for item in rec_res_all:
            file.write(item + "\n")
        file.close()


if __name__ == "__main__":
    # parse args
    args = parse_args()
    set_logger(name="mindocr")
    save_dir = args.draw_img_save_dir
    img_paths = get_image_paths(args.image_dir)

    ms.set_context(mode=args.mode)

    # init detector
    text_recognize = TextRecognizer(args)
    start = time()
    rec_res_all = text_recognize(img_paths, do_visualize=False)
    t = time() - start

    # save all results in a txt file
    save_fp = os.path.join(save_dir, "formula_results.txt")
    save_rec_res(rec_res_all, save_path=save_fp)
    logger.info(f"All rec res: {rec_res_all}")
    logger.info(f"Done! Text recognition results saved in {save_dir}")
    logger.info(f"Time cost: {t}, FPS: {len(img_paths) / t}")
