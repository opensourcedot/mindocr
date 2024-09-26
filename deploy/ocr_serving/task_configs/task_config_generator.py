import os

import yaml

from deploy.ocr_serving.package_utils.path_utils import get_base_path, bfs_search_specific_type_file

COMBINED_CONFIG_YAML_FILE_NAME = "combined_configs.yaml"
MODELS_LINK_PATH = "deploy/ocr_serving/task_configs/model_link"


class TaskConfigGenerator:
    def __init__(self):
        self.__mindocr_base_path = get_base_path()
        self.__models_link_path = os.path.join(self.__mindocr_base_path, MODELS_LINK_PATH)
        self.__models_link = []

    def __get_key_information_from_yaml(self, yaml_file_path: str) -> dict:
        """

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

    def get_models_link(self):
        with open(self.__models_link_path, "r", encoding="utf-8") as f:
            models_link = yaml.load_all(f, Loader=yaml.FullLoader)


    def combine_all_yaml_files_key_info(self):
        """
        get all_configs.yaml
        Returns:
        """
        configs_base_path = os.path.join(self.__mindocr_base_path, "configs")

        final_result = []
        config_yaml_files = bfs_search_specific_type_file(configs_base_path, ".yaml")
        for yaml_file in config_yaml_files:
            final_result.append(self.__get_key_information_from_yaml(yaml_file))
        with open("all_configs.yaml", "w+", encoding="utf-8") as f:
            yaml.dump_all(documents=final_result, stream=f, allow_unicode=True)


if __name__ == "__main__":
    task_config_generator = TaskConfigGenerator()
    task_config_generator.combine_all_yaml_files_key_info()
