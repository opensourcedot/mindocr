import copy
import os

import yaml

from deploy.ocr_serving.package_utils.path_utils import get_base_path, bfs_search_specific_type_file

COMBINED_CONFIG_YAML_FILE_NAME = "combined_configs.yaml"
MODELS_LINK_PATH = "deploy/ocr_serving/task_configs/model_link.yaml"


class TaskConfigGenerator:
    def __init__(self):
        self.__mindocr_base_path = get_base_path()
        self.__models_link_path = os.path.join(self.__mindocr_base_path, MODELS_LINK_PATH)
        self.__models_link = []
        self.__config_yaml_files = []
        self.__model_configs = []

    def __get_key_information_from_yaml(self, yaml_file_path: str) -> dict:
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
            pre_process = result["eval"]["dataset"]["transform_pipeline"] if "eval" in result.keys() else None
            config_dict = {"postprocess": post_process, "preprocess": pre_process, "use_pretrained_mindir": True,
                           "yaml_file_name": os.path.basename(yaml_file_path)}
        return config_dict

    def __get_models_link(self):
        """
        get models link(ckpt link / mindir link)
        Returns:
        """
        with open(self.__models_link_path, "r", encoding="utf-8") as f:
            models_link = yaml.load_all(f, Loader=yaml.FullLoader)
            for model_link in models_link:
                self.__models_link.append(model_link)

    def __add_model_link_to_model_configs(self):
        """
        add model link(ckpt/ mindir) to model configs
        Returns:
        """
        processed_model_configs = []
        for model_config in self.__model_configs:
            model_config["ckpt_link"] = ""
            model_config["mindir_link"] = ""
            for model_link in self.__models_link:
                if model_config["yaml_file_name"] == model_link["config_yaml_path"]:
                    model_config["ckpt_link"] = model_link["ckpt_link"]
                    model_config["mindir_link"] = model_link["mindir_link"]
                    break
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
            self.__model_configs.append(self.__get_key_information_from_yaml(yaml_file))

        # 4. get model links (include ckpt link, mindir link)
        self.__get_models_link()

        # 5. add model ckpt/ mindir link to model_configs
        self.__add_model_link_to_model_configs()

        # 4. combine these configs into one config.yaml
        with open("all_configs.yaml", "w+", encoding="utf-8") as f:
            yaml.dump_all(documents=self.__model_configs, stream=f, allow_unicode=True)


if __name__ == "__main__":
    task_config_generator = TaskConfigGenerator()
    task_config_generator.generate_config_file()
