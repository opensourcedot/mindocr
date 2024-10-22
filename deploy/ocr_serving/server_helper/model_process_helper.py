import os
import sys
from os.path import dirname

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

    @property
    def postprocess_method(self):
        return build_postprocess(self.related_yaml_path)
