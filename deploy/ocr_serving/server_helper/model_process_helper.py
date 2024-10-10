import copy

import yaml

from mindocr.data.transforms import create_transforms
from mindocr.postprocess import build_postprocess


class ModelProcessor:
    def __init__(self, related_yaml_path: str):
        self.related_yaml_path = related_yaml_path
        self.yaml_config = None
        self.__read_yaml()

    def __read_yaml(self):
        with open(self.related_yaml_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.yaml_config = copy.deepcopy(yaml_config)

    @property
    def preprocess_method(self):
        transforms = create_transforms(self.yaml_config["preprocess"])
        for transform in transforms:
            if "DecodeImage" in transform:
                transform["DecodeImage"].update({"keep_ori": True})
            break
        return transforms

    @property
    def postprocess_method(self):
        return build_postprocess(self.yaml_config["postprocess"])
