import os
import sys
from os.path import dirname
from typing import List, Tuple

import numpy as np
from mindspore_serving.server import register

from .model_process_helper import ModelProcessor

current_file_path = os.path.abspath(__file__)
mindocr_path = dirname(dirname(dirname(dirname(current_file_path))))
if mindocr_path not in sys.path:
    sys.path.append(mindocr_path)

model_processor = ModelProcessor(os.path.join(dirname(current_file_path), "config.yaml"))


# define preprocess and postprocess
def preprocess(data_nparray: np.ndarray) -> tuple:
    return model_processor.preprocess(data_nparray)


def postprocess(preds_0, preds_1, preds_2) -> List[np.ndarray]:
    return model_processor.postprocess_method([preds_0, preds_1, preds_2])["polys"]


# register model
model = register.declare_model(model_file="model.mindir", model_format="MindIR", with_batch_dim=False)


def model_infer(net_inputs) -> Tuple:
    preds_0, preds_1, preds_2 = model.call(net_inputs)
    return preds_0, preds_1, preds_2


# register url
@register.register_method(output_names=["polys"])
def infer(image):
    net_inputs = register.add_stage(preprocess, image, outputs_count=1)
    preds_0, preds_1, preds_2 = register.add_stage(model_infer, net_inputs, outputs_count=3)
    polys = register.add_stage(postprocess, preds_0, preds_1, preds_2, outputs_count=1)
    return polys