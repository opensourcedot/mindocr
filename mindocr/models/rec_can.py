from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel

__all__ = ['CAN', 'can']


def _cfg(url="", **kwargs):
    return {"url": url, **kwargs}


default_cfgs = {
    'can': _cfg(
        url='https://download-mindspore.osinfra.cn/model_zoo/research/cv/can/can_params.ckpt'),
    }


class CAN(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def can(pretrained=False, **kwargs):
    model_config = {
            "backbone": {
                "name": "rec_can_densenet",
                "pretrained": False,
                "growth_rate": 24, 
                "reduction": 0.5, 
                "bottleneck": True, 
                "use_dropout": True,
                "input_channels": 1,
                },
            "head": {
                "name": "CANHead",
                "out_channels": 111,
                "ratio": 16,
                "attdecoder": {
                    "is_train": False,
                    "input_size": 256,
                    "hidden_size": 256,
                    "encoder_out_channel": 684,
                    "dropout": True,
                    "dropout_ratio": 0.5,
                    "word_num": 111,
                    "counting_decoder_out_channel": 111,
                    "attention": {
                            "attention_dim": 512,
                            "word_conv_kernel": 1,
                        },
                    },
                },
    }
    model = CAN(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['can']
        load_pretrained(model, default_cfg)

    return model
