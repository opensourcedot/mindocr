import os
import yaml

from mindocr.data.transforms import create_transforms, run_transforms

__dir__ = os.path.dirname(os.path.abspath(__file__))
__all__ = ["Preprocessor"]


class Preprocessor(object):
    def __init__(self, cfg_source="default", **kwargs):
        if cfg_source == "yaml":
            yaml_cfg = kwargs["yaml_preproc_cfg"]
            preproc_cfg = self._get_yaml_cfg(yaml_cfg)
        else:
            preproc_cfg = self._get_default_cfg(**kwargs)

        self.preproc_cfg = preproc_cfg
        self.transforms = create_transforms(preproc_cfg)

    def __call__(self, img_or_path):
        """
        Return:
            dict, preprocessed data containing keys:
                - image: np.array, transfomred image
                - image_ori: np.array, original image
                - shape: list of [ori_h, ori_w, scale_h, scale_w]
                and other keys added in transform pipeline.
        """
        if isinstance(img_or_path, str):
            data = {"img_path": img_or_path}
            output = run_transforms(data, self.transforms)
        elif isinstance(img_or_path, dict):
            output = run_transforms(img_or_path, self.transforms)
        else:
            data = {"image": img_or_path}
            data["image_ori"] = img_or_path.copy()  # TODO
            data["image_shape"] = img_or_path.shape
            output = run_transforms(data, self.transforms[1:])

        return output

    def _get_default_cfg(self, **kwargs):
        task = kwargs.get("task")
        algo = kwargs.get("algo")
        config_name = "default_cfg/preprocess.yaml"
        config_path = "{}/{}".format(__dir__, config_name)

        preproc_cfg = []
        with open(config_path, "r") as f:
            preproc_cfg = yaml.safe_load(f)[task][algo]

        if task == "rec" and algo in {"CRNN", "CRNN_CH", "RARE", "RARE_CH", "SVTR", "SVTR_PPOCRv3_CH"}:
            batch_mode = kwargs.get("rec_batch_mode", False)
            if not batch_mode:
                preproc_cfg[1]["RecResizeNormForInfer"]["padding"] = False
                preproc_cfg[1]["RecResizeNormForInfer"]["keep_ratio"] = True
                preproc_cfg[1]["RecResizeNormForInfer"]["target_width"] = None

        return preproc_cfg

    def _get_yaml_cfg(self, yaml_cfg: list):
        filtered_cfg = [d for d in yaml_cfg if not any("label" in key.lower() for key in d.keys())]

        return filtered_cfg


if __name__ == "__main__":
    pre = Preprocessor("det")
