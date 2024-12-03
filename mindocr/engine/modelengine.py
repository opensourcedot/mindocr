import yaml

from .postprocess import Postprocessor
from .preprocess import Preprocessor
from .utils import *
from ..models import build_model

__all__ = ["ModelEngine"]


class ModelEngine(object):
    def __init__(self, init_with_config_file=False, **kwargs):
        self.init_with_config_file = init_with_config_file
        if init_with_config_file:
            assert (
                kwargs["config_file_path"] is not None
            ), f"Init params by yaml, but the config_file_path is None"
            self.parse_config_from_yaml(kwargs["config_file_path"])
        else:
            self.kwargs = kwargs
        self.preprocess()
        self.postprocess()
        self.model = self.load_model()

    def preprocess(self):
        if self.init_with_config_file:
            self.preprocess = Preprocessor(
                cfg_source = "yaml", 
                task = self.task,
                yaml_preproc_cfg = self.yaml_preproc_cfg
            )
        else:
            self.preprocess = Preprocessor(
                cfg_source = "default", 
                **self.kwargs
            )

    def postprocess(self):
        if self.init_with_config_file:
            self.postprocess = Postprocessor(  
                cfg_source = "yaml", 
                task = self.task,
                yaml_postproc_cfg = self.yaml_postproc_cfg
            )
        else:
            self.postprocess = Postprocessor(
                cfg_source = "default", 
                **self.kwargs
            )

    def load_model(self, **kwargs):
        if self.init_with_config_file:
            self.model = build_model(
                self.model_cfg,
                pretrained = True,
                ckpt_load_path=self.ckpt_load_path,
                amp_level=self.amp_level
            )
        else:
            self.model_dir = self.kwargs.get("model_dir")
            self.model_name = self.kwargs.get("model_name")
            self.amp_level = self.kwargs.get("amp_level")
            if self.model_dir is None:
                pretrained = True
                ckpt_load_path = None
            else:
                pretrained = False
                ckpt_load_path = get_ckpt_file(self.model_dir)

            self.model = build_model(
                self.model_name,
                pretrained=pretrained,
                ckpt_load_path=ckpt_load_path,
                amp_level=self.amp_level,
            )
        
        return self.model

    def get_model(self):
        return self.model

    def parse_config_from_yaml(self,config_file_path):
        with open(config_file_path) as f:
            all_cfg = yaml.safe_load(f)
            self.yaml_preproc_cfg: list = all_cfg["eval"]["dataset"]["transform_pipeline"]
            self.yaml_postproc_cfg: dict = all_cfg["postprocess"]
            self.model_cfg = all_cfg["model"]
            self.task = all_cfg["model"]["type"]
            self.amp_level = all_cfg["system"]["amp_level"]
            self.ckpt_load_path = all_cfg["eval"]["ckpt_load_path"]
