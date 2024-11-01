import os
import sys

import pandas as pd
import yaml

current_file_path = os.path.abspath(__file__)
mindocr_path = os.path.dirname(os.path.dirname(current_file_path))

if mindocr_path not in sys.path:
    sys.path.append(mindocr_path)

from package_utils.path_utils import bfs_search_specific_type_file, get_base_path

MODELS_LINK_PATH = "deploy/ocr_serving/task_configs/model_link_mapper.csv"
OUTPUT_CONFIGS_SAVE_PATH = os.path.join(get_base_path(), "deploy/ocr_serving/task_configs/all_configs.yaml")


def get_key_information_from_yaml(yaml_file_path: str) -> dict:
    """
    get postprocess, preprocess, yaml_file_name from yanl
    Args:
        yaml_file_path: yaml_file_path

    Returns:
        full key information of all yaml files in configs of mindocr
    """
    with open(yaml_file_path, "r", encoding="utf-8") as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
        post_process = result["postprocess"] if "postprocess" in result.keys() else None
        eval = result["eval"] if "eval" in result.keys() else None
        pre_process = result["eval"]["dataset"]["transform_pipeline"] if "predict" in result.keys() else None
        config_dict = {
            "postprocess": post_process,
            "preprocess": pre_process,
            "use_pretrained_mindir": True,
            "yaml_file_name": os.path.basename(yaml_file_path),
            "eval": eval,
        }
    return config_dict


class TaskConfigGenerator:
    def __init__(self):
        self.__mindocr_base_path = get_base_path()
        self.__models_link_path = os.path.join(self.__mindocr_base_path, MODELS_LINK_PATH)
        self.__models_link = []
        self.__config_yaml_files = []
        self.__model_configs = []

    def __get_models_links(self):
        """
        from markdown file to get models links
        Returns:
        """
        path = os.path.join(self.__mindocr_base_path, MODELS_LINK_PATH)
        model_link_df = pd.read_csv(path)
        for ind in model_link_df.index:
            info_dict = model_link_df.loc[ind].to_dict()
            info_dict["data_shape_nchw"] = info_dict["data_shape_nchw"][1:-1].split(",")
            self.__models_link.append(info_dict)

    def __add_model_link_to_model_configs(self):
        """
        add model link(ckpt/ mindir) to model configs
        Returns:
        """
        processed_model_configs = []
        for model_config in self.__model_configs:
            model_config["ckpt_link"] = ""
            model_config["custom_mindir_link"] = ""
            for model_link in self.__models_link:
                if model_config["yaml_file_name"] == model_link["yaml_file_name"]:
                    model_config["ckpt_link"] = model_link["ckpt_link"]
                    model_config["data_shape_nchw"] = model_link["data_shape_nchw"]
                    break
            if not model_config["ckpt_link"]:
                model_config["use_pretrained_mindir"] = False
            processed_model_configs.append(model_config)
        self.__model_configs = processed_model_configs

    def generate_config_file(self):
        """
        get all_configs.yaml
        Returns:
        """
        # 1. set base path (mindocr/ configs)
        configs_base_path = os.path.join(self.__mindocr_base_path, "configs")

        # 2. search config yaml files from base path
        config_yaml_files = bfs_search_specific_type_file(configs_base_path, ".yaml")

        # 3. get model preprocess, postprocess ... from model config yaml files
        for yaml_file in config_yaml_files:
            self.__model_configs.append(get_key_information_from_yaml(yaml_file))

        # 4. get model links (include ckpt link, mindir link)
        self.__get_models_links()

        # 5. add model ckpt/ mindir link to model_configs
        self.__add_model_link_to_model_configs()

        # 6. combine these configs into one config.yaml
        with open(OUTPUT_CONFIGS_SAVE_PATH, "w+", encoding="utf-8") as f:
            yaml.dump_all(documents=self.__model_configs, stream=f, allow_unicode=True)


if __name__ == "__main__":
    task_config_generator = TaskConfigGenerator()
    task_config_generator.generate_config_file()
