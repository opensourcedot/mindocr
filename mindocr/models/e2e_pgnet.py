from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel

__all__ = ['E2eNet', 'pgnet_resnet50']


def _cfg(url="", **kwargs):
    return {"url": url, **kwargs}


default_cfgs = {
    "PGNet": _cfg(
        url="https://download-mindspore.osinfra.cn/model_zoo/research/cv/pgnet/pgnet_best_weight.ckpt"
    ),
}


class E2eNet(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def pgnet_resnet50(pretrained=False, pretrained_backbone=False, **kwargs):
    model_config = {
        "backbone": {
            'name': 'pgnet_backbone',
            'pretrained': pretrained_backbone
        },
        "neck": {
            "name": 'E2eFpn'
        },
        "head": {
            "name": 'PGNetHead'
        }
    }
    model = E2eNet(model_config)

    if pretrained:
        default_cfg = default_cfgs['PGNet']
        load_pretrained(model, default_cfg)
    return model
