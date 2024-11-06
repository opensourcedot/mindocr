import os
import sys
import cv2
import json
import logging
import time

from typing import Dict, List, Tuple

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor


from predict_layout import LayoutAnalyzer
from predict_system import TextSystem
from predict_table import TableAnalyzer
from config import create_parser, str2bool
from utils import get_ckpt_file, get_image_paths, convert_info_docx, sorted_layout_boxes

from mindocr.utils.logger import set_logger

from utils import *

logger = logging.getLogger("mindocr")

def e2e_parse_args():
    '''
    Inherit the parser from the config.py file, and add the following arguments:
        1. layout: Whether to enable layout analyzer
        2. ocr: Whether to enable ocr
        3. table: Whether to enable table recognizer
        4. recovery: Whether to recovery output to docx
    '''
    parser = create_parser()

    parser.add_argument(
        "--layout",
        type=str2bool,
        default=False,
        help="Whether to enable layout analyzer.",
    )

    parser.add_argument(
        "--ocr",
        type=str2bool,
        default=False,
        help="Whether to enable ocr.",
    )

    parser.add_argument(
        "--table",
        type=str2bool,
        default=False,
        help="Whether to table recognizer.",
    )

    parser.add_argument(
        "--recovery",
        type=str2bool,
        default=False,
        help="Whether to recovery output to docx.",
    )

    args = parser.parse_args()
    return args

def init_ocr(args):
    '''
    Initialize text detection and recognition system

    Args:
        ocr: enable text system or not
        det_algorithm: detection algorithm
        rec_algorithm: recognition algorithm
        det_model_dir: detection model directory
        rec_model_dir: recognition model directory
    '''
    if args.ocr:
        args.det_algorithm = "DB++"
        args.rec_algorithm = "SVTR_PPOCRv3_CH"
        return TextSystem(args)
    return None 

def init_layout(args):
    '''
    Initialize layout analysis system

    Args:
        layout: enable layout module or not
        layout_algorithm: layout algorithm
        layout_model_dir: layout model ckpt path
        layout_amp_level: Auto Mixed Precision level for layout
    '''
    if args.layout:
        return LayoutAnalyzer(args)
    return None

def init_table(args):
    '''
    Initialize table recognition system

    Args:
        table: enable table recognizer or not
        table_algorithm: table algorithm
        table_model_dir: table model ckpt path
        table_max_len: max length of the input image
        table_char_dict_path: path to character dictionary for table
        table_amp_level: Auto Mixed Precision level for table
    '''
    if args.table:
        args.det_algorithm = "DB_PPOCRv3"
        args.rec_algorithm = "SVTR_PPOCRv3_CH"
        return TableAnalyzer(args)
    return None

def predict_e2e():
    #set_logger(name="mindocr")
    first_time = time.time()
    args = e2e_parse_args()
    save_folder = args.draw_img_save_dir

    save_folder, _ = os.path.splitext(save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    text_system = init_ocr(args)
    layout_analyzer = init_layout(args)
    table_analyzer = init_table(args)

    img_paths = get_image_paths(args.image_dir)

    for index, img_path in enumerate(img_paths):
        img_name = os.path.basename(img_path).rsplit(".", 1)[0]
        image = cv2.imread(img_path)

        if args.layout:
            results = layout_analyzer(image, do_visualize=args.visualize_output)
        else:
            results = [{"category_id": 1, "bbox": [0, 0, image.shape[1], image.shape[0]], "score": 1.0}]

        # crop text regions
        h_ori, w_ori = image.shape[:2]
        category_dict = {1: 'text', 2: 'title', 3: 'list', 4: 'table', 5: 'figure'}
        final_results = []
        for i in range(len(results)):
            category_id = results[i]['category_id']
            left, top, w, h = results[i]['bbox']
            right = left + w
            bottom = top + h
            cropped_img = image[int(top):int(bottom), int(left):int(right)]

            if (category_id == 1 or category_id == 2 or category_id == 3) and args.ocr:
                start_time = time.time()

                # only add padding for text, title and list images for better recognition
                cropped_img = add_padding(cropped_img, padding_size=10, padding_color=(255, 255, 255))

                rec_res_all_crops = text_system(cropped_img, do_visualize=args.visualize_output)
                output = sort_words_by_poly(rec_res_all_crops[1], rec_res_all_crops[0])
                final_results.append({"type": category_dict[category_id], "bbox": [left, top, right, bottom],
                    "res": " ".join(output)})

                logger.info(
                    f"Processing {category_dict[category_id]} at [{left}, {top}, {right}, {bottom}]"
                        f" {time.time() - start_time:.2f}s"
                )
            elif category_id == 4 and args.table:
                table_start_time = time.time()
                pred_html, _ = table_analyzer(cropped_img, do_visualize=args.visualize_output)
                final_results.append({"type": category_dict[category_id], "bbox": [left, top, right, bottom],
                    "res": pred_html})

                logger.info(
                    f"Processing {category_dict[category_id]} at [{left}, {top}, {right}, {bottom}]"
                        f" {time.time() - start_time:.2f}s"
                )
            else:
                save_path = save_folder + f"{img_name}_figure_{i}.png"
                cv2.imwrite(save_path, cropped_img)
                final_results.append({"type": category_dict[category_id], "bbox": [left, top, right, bottom],
                    "res": save_path})
        
        if args.recovery:
            final_results = sorted_layout_boxes(final_results, w_ori)
            convert_info_docx(final_results, save_folder, f"{img_name}_converted_docx")
    
    logger.info(f"Processing e2e total time: {time.time() - first_time:.2f}s")
    logger.info(f"Done! predict {len(img_paths)} e2e results saved in {save_folder}")

if __name__ == "__main__":
    predict_e2e()
