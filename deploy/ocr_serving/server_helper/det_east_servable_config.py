import os
import sys
from io import BytesIO
from os.path import dirname
from typing import List

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
    return result["net_inputs"]


def postprocess(scores, geo) -> List[np.ndarray]:
    return model_processor.postprocess_method(tuple((scores, geo)))["polys"]


# register model
model = register.declare_model(model_file="model.mindir", model_format="MindIR", with_batch_dim=False)


def infer(net_inputs) -> tuple:
    scores, geo = model.call(net_inputs)
    return scores, geo


# register url
@register.register_method(output_names=["polys"])
def det_infer(image):
    net_inputs = register.add_stage(preprocess, image, outputs_count=1)
    scores, geo = register.add_stage(infer, net_inputs, outputs_count=2)
    polys = register.add_stage(postprocess, scores, geo, outputs_count=1)
    return polys
