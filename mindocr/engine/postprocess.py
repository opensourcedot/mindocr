import os
import sys
import yaml
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from mindocr.postprocess import build_postprocess


class Postprocessor(object):
    def __init__(self, cfg_source="default", **kwargs):

        if cfg_source == "yaml":
            yaml_cfg = kwargs["yaml_postproc_cfg"]
            postproc_cfg = self._get_yaml_cfg(yaml_cfg)
        else:
            postproc_cfg = self._get_default_cfg(**kwargs)

        self.task = kwargs.get("task")
        self.postprocess = build_postprocess(postproc_cfg)

    def __call__(self, pred, data=None, **kwargs):
        """
        Args:
            pred: network prediction
            data: (optional)
                preprocessed data, dict, which contains key `shape`
                    - shape: its values are [ori_img_h, ori_img_w, scale_h, scale_w]. scale_h, scale_w are needed to
                      map the predicted polygons back to the orignal image shape.

        return:
            det_res: dict, elements:
                    - polys: shape [num_polys, num_points, 2], point coordinate definition: width (horizontal),
                      height(vertical)
        """

        if self.task == "det":
            if self.rescale_internally:
                shape_list = np.array(data["shape_list"], dtype="float32")
                shape_list = np.expand_dims(shape_list, axis=0)
            else:
                shape_list = None

            output = self.postprocess(pred, shape_list=shape_list)

            if isinstance(output, dict):
                polys = output["polys"][0]
                scores = output["scores"][0]
            else:
                polys, scores = output[0]

            if not self.rescale_internally:
                scale_h, scale_w = data["shape_list"][2:]
                if len(polys) > 0:
                    if not isinstance(polys, list):
                        polys[:, :, 0] = polys[:, :, 0] / scale_w
                        polys[:, :, 1] = polys[:, :, 1] / scale_h
                        if self.round:
                            polys = np.round(polys)
                    else:
                        for i, poly in enumerate(polys):
                            polys[i][:, 0] = polys[i][:, 0] / scale_w
                            polys[i][:, 1] = polys[i][:, 1] / scale_h
                            if self.round:
                                polys[i] = np.round(polys[i])

            det_res = dict(polys=polys, scores=scores)

            return det_res
        elif self.task == "rec":
            output = self.postprocess(pred)
            return output
        elif self.task == "ser":
            output = self.postprocess(
                pred, segment_offset_ids=kwargs.get("segment_offset_ids"), ocr_infos=kwargs.get("ocr_infos")
            )
            return output
    
    def _get_default_cfg(self, **kwargs):

        task = kwargs.get("task")
        algo = kwargs.get("algo")

        if task=="det":
            self.rescale_internally = True
            self.round = True
        config_name = "default_cfg/postprocess.yaml"
        config_path = "{}/{}".format(__dir__, config_name) 

        postproc_cfg = {}
        with open(config_path, "r") as f:
            postproc_cfg = yaml.safe_load(f)[task][algo]      

        return postproc_cfg

    def _get_yaml_cfg(self, yaml_cfg: dict):

        return yaml_cfg


if __name__=='__main__':
    pre = Postprocessor()
