import os
import sys
from io import BytesIO
from os.path import dirname
from typing import Dict, List

import numpy as np
from PIL import Image
from mindspore_serving.server import register

from .model_process_helper import ModelProcessor

current_file_path = os.path.abspath(__file__)
mindocr_path = dirname(dirname(dirname(dirname(current_file_path))))
if mindocr_path not in sys.path:
    sys.path.append(mindocr_path)

model_processor = ModelProcessor(os.path.join(dirname(current_file_path), "config.yaml"))


# define preprocess and postprocess
def preprocess(data_nparray: np.ndarray) -> tuple:
    image = Image.open(BytesIO(data_nparray.tobytes()))
    result = model_processor.preprocess_method([np.array(image)])
    return result["net_inputs"], result["shape_list"]


def postprocess(scores: np.ndarray, geo: np.ndarray, shape_list: np.ndarray) -> List[np.ndarray]:
    polys = model_processor.postprocess_method((scores, geo), shape_list)["polys"][0]
    polys = [np.array(x) for x in polys]
    return [polys]


# register model
model = register.declare_model(model_file="model.mindir", model_format="MindIR", with_batch_dim=False)


def infer(net_inputs) -> tuple:
    scores, geo = model.call(net_inputs)
    return scores, geo


# register url
@register.register_method(output_names=["polys"])
def det_infer(image):
    net_inputs, shape_list = register.add_stage(preprocess, image, outputs_count=2)
    scores, geo = register.add_stage(infer, net_inputs, outputs_count=2)
    polys = register.add_stage(postprocess, scores, geo, shape_list, outputs_count=1)
    return polys
