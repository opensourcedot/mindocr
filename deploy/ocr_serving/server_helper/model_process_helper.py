import os
import sys
from os.path import dirname
from typing import Tuple
from io import BytesIO

import numpy as np
from PIL import Image


current_file_path = os.path.abspath(__file__)
mindocr_path = dirname(dirname(dirname(dirname(dirname(current_file_path)))))
if mindocr_path not in sys.path:
    sys.path.append(mindocr_path)
from deploy.py_infer.src.data_process.preprocess.builder import build_preprocess
from deploy.py_infer.src.data_process.postprocess.builder import build_postprocess


class ModelProcessor:
    def __init__(self, related_yaml_path: str):
        self.related_yaml_path = related_yaml_path

    @property
    def preprocess_method(self):
        return build_preprocess(self.related_yaml_path, False)

    def preprocess(self, data_nparray: np.ndarray) -> Tuple:
        image = Image.open(BytesIO(data_nparray.tobytes()))
        result = self.preprocess_method([np.array(image)])
        return result["net_inputs"]

    @property
    def postprocess_method(self):
        return build_postprocess(self.related_yaml_path)
