from typing import List, Union

import cv2
import numpy as np
from PIL import Image

from mindspore import dataset as ds
from mindspore.dataset import vision

from ...data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

__all__ = [
    "DecodeImage",
    "NormalizeImage",
    "ToCHWImage",
    "PackLoaderInputs",
    "RandomScale",
    "RandomColorAdjust",
    "RandomRotate",
    "RandomHorizontalFlip",
    "CANImageNormalize",
    "GrayImageChannelFormat",
]


def get_value(val, name):
    if isinstance(val, str) and val.lower() == "imagenet":
        assert name in ["mean", "std"]
        return IMAGENET_DEFAULT_MEAN if name == "mean" else IMAGENET_DEFAULT_STD
    elif isinstance(val, list):
        return val
    else:
        raise ValueError(f"Wrong {name} value: {val}")


class DecodeImage:
    """
    img_mode (str): The channel order of the output, 'BGR' and 'RGB'. Default to 'BGR'.
    channel_first (bool): if True, image shpae is CHW. If False, HWC. Default to False
    """

    def __init__(
        self, img_mode="BGR", channel_first=False, to_float32=False, ignore_orientation=False, keep_ori=False, **kwargs
    ):
        self.img_mode = img_mode
        self.to_float32 = to_float32
        self.channel_first = channel_first
        self.flag = cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR if ignore_orientation else cv2.IMREAD_COLOR
        self.keep_ori = keep_ori

        self.use_minddata = kwargs.get("use_minddata", False)
        self.decoder = None
        self.cvt_color = None
        if self.use_minddata:
            self.decoder = vision.Decode()
            self.cvt_color = vision.ConvertColor(vision.ConvertMode.COLOR_BGR2RGB)

    def __call__(self, data: dict):
        if "img_path" in data:
            with open(data["img_path"], "rb") as f:
                img = f.read()
        elif "img_lmdb" in data:
            img = data["img_lmdb"]
        elif "np_format_img" in data:
            img = data["np_format_img"]
        else:
            raise ValueError('"img_path" or "img_lmdb" must be in input data')

        if "np_format_img" not in data:
            img = np.frombuffer(img, dtype="uint8")

        if self.use_minddata:
            img = self.decoder(img)
            if self.img_mode == "BGR":
                img = self.cvt_color(img)
        else:
            img = cv2.imdecode(img, self.flag)
            if self.img_mode == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        if self.to_float32:
            img = img.astype("float32")
        data["image"] = img
        # data['ori_image'] = img.copy()
        data["raw_img_shape"] = img.shape[:2]

        if self.keep_ori:
            data["image_ori"] = img.copy()
        return data


class NormalizeImage:
    """
    normalize image, subtract mean, divide std
    input image: by default, np.uint8, [0, 255], HWC format.
    return image: float32 numpy array
    """

    def __init__(
        self,
        mean: Union[List[float], str] = "imagenet",
        std: Union[List[float], str] = "imagenet",
        is_hwc=True,
        bgr_to_rgb=False,
        rgb_to_bgr=False,
        **kwargs,
    ):
        # By default, imagnet MEAN and STD is in RGB order. inverse if input image is in BGR mode
        self._channel_conversion = False
        if bgr_to_rgb or rgb_to_bgr:
            self._channel_conversion = True

        # TODO: detect hwc or chw automatically
        shape = (3, 1, 1) if not is_hwc else (1, 1, 3)
        self.mean = get_value(mean, "mean")
        self.std = get_value(std, "std")
        self.is_hwc = is_hwc

        self.use_minddata = kwargs.get("use_minddata", False)
        self.normalize = None
        self.cvt_color = None
        if self.use_minddata:
            self.decoder = vision.Normalize(self.mean, self.std, is_hwc)
            self.cvt_color = vision.ConvertColor(vision.ConvertMode.COLOR_BGR2RGB)
        else:
            self.mean = np.array(self.mean).reshape(shape).astype("float32")
            self.std = np.array(self.std).reshape(shape).astype("float32")

    def __call__(self, data):
        img = data["image"]
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"

        if self.use_minddata:
            if self._channel_conversion:
                img = self.cvt_color(img)
            img = self.normalize(img)
            data["image"] = img
            return data

        if self._channel_conversion:
            if self.is_hwc:
                img = img[..., [2, 1, 0]]
            else:
                img = img[[2, 1, 0], ...]

        data["image"] = (img.astype("float32") - self.mean) / self.std
        return data


class ToCHWImage:
    # convert hwc image to chw image
    def __init__(self, **kwargs):
        self.use_minddata = kwargs.get("use_minddata", False)
        self.hwc2chw = None
        if self.use_minddata:
            self.hwc2chw = vision.HWC2CHW()

    def __call__(self, data):
        img = data["image"]
        if isinstance(img, Image.Image):
            img = np.array(img)
        if self.use_minddata:
            data["image"] = self.hwc2chw(img)
            return data
        data["image"] = img.transpose((2, 0, 1))
        return data


class PackLoaderInputs:
    """
    Args:
        output_columns (list): the keys in data dict that are expected to output for dataloader

    Call:
        input: data dict
        output: data tuple corresponding to the `output_columns`
    """

    def __init__(self, output_columns: List, **kwargs):
        self.output_columns = output_columns

    def __call__(self, data):
        out = []
        for k in self.output_columns:
            assert k in data, f"key {k} does not exists in data, availabe keys are {data.keys()}"
            out.append(data[k])

        return tuple(out)


class RandomScale:
    """
    Randomly scales an image and its polygons in a predefined scale range.
    Args:
        scale_range: (min, max) scale range.
        size_limits: (min_side_len, max_side_len) size limits. Default: None.
        p: probability of the augmentation being applied to an image.
    """

    def __init__(
        self,
        scale_range: Union[tuple, list],
        size_limits: Union[tuple, list] = None,
        p: float = 0.5,
        **kwargs,
    ):
        self._range = sorted(scale_range)
        self._size_limits = sorted(size_limits) if size_limits else []
        self._p = p
        assert kwargs.get("is_train", True), ValueError("RandomScale augmentation must be used for training only")

    def __call__(self, data: dict) -> dict:
        """
        required keys:
            image, HWC
            (polys)
        modified keys:
            image
            (polys)
        """
        if np.random.random() < self._p:
            if self._size_limits:
                size = data["image"].shape[:2]
                min_scale = max(self._size_limits[0] / size[0], self._size_limits[0] / size[1], self._range[0])
                max_scale = min(self._size_limits[1] / size[0], self._size_limits[1] / size[1], self._range[1])
                scale = np.random.uniform(min_scale, max_scale)
            else:
                scale = np.random.uniform(*self._range)

            data["image"] = cv2.resize(data["image"], dsize=None, fx=scale, fy=scale)
            if "polys" in data:
                data["polys"] *= scale

        return data


class RandomColorAdjust:
    def __init__(self, brightness=32.0 / 255, saturation=0.5, **kwargs):
        contrast = kwargs.get("contrast", (1, 1))
        hue = kwargs.get("hue", (0, 0))
        self._jitter = vision.RandomColorAdjust(
            brightness=brightness, saturation=saturation, contrast=contrast, hue=hue
        )
        self._jitter.implementation = ds.Implementation.C

    def __call__(self, data):
        """
        required keys: image
        modified keys: image
        """
        # there's a bug in MindSpore that requires images to be converted to the PIL format first
        data["image"] = self._jitter(data["image"])
        return data


class RandomRotate:
    """
    Randomly rotate an image with polygons in it (if any).
    Args:
        degrees: range of angles [min, max]
        expand_canvas: whether to expand canvas during rotation (the image size will be increased) or
                       maintain the original size (the rotated image will be cropped back to the original size).
        p: probability of the augmentation being applied to an image.
    """

    def __init__(self, degrees=(-10, 10), expand_canvas=True, p: float = 1.0, **kwargs):
        self._degrees = degrees
        self._canvas = expand_canvas
        self._p = p

    def __call__(self, data: dict) -> dict:
        if np.random.random() < self._p:
            angle = np.random.randint(self._degrees[0], self._degrees[1])
            h, w = data["image"].shape[:2]

            center = w // 2, h // 2  # x, y
            mat = cv2.getRotationMatrix2D(center, angle, 1)

            if self._canvas:
                # compute the new bounding dimensions of the image
                cos, sin = np.abs(mat[0, 0]), np.abs(mat[0, 1])
                w, h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))

                # adjust the rotation matrix to take into account translation
                mat[0, 2] += (w / 2) - center[0]
                mat[1, 2] += (h / 2) - center[1]

            data["image"] = cv2.warpAffine(data["image"], mat, (w, h))

            if "polys" in data and len(data["polys"]):
                data["polys"] = cv2.transform(data["polys"], mat)

        return data


class RandomHorizontalFlip:
    """
    Random horizontal flip of an image with polygons in it (if any).
    Args:
        p: probability of the augmentation being applied to an image.
    """

    def __init__(self, p: float = 0.5, **kwargs):
        self._p = p

    def __call__(self, data: dict) -> dict:
        if np.random.random() < self._p:
            data["image"] = cv2.flip(data["image"], 1)

            if "polys" in data and len(data["polys"]):
                mat = np.float32([[-1, 0, data["image"].shape[1] - 1], [0, 1, 0]])
                data["polys"] = cv2.transform(data["polys"], mat)
                # TODO: assign a new starting point located in the top left
                data["polys"] = data["polys"][:, ::-1, :]  # preserve the original order (e.g. clockwise)

        return data


class CANImageNormalize(object):
    """normalize image such as substract mean, divide std"""

    def __init__(self, scale=None, mean=None, std=None, order="chw", **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == "chw" else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype("float32")
        self.std = np.array(std).reshape(shape).astype("float32")

    def __call__(self, data):
        img = data["image"]
        from PIL import Image

        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
        data["image"] = (img.astype("float32") * self.scale - self.mean) / self.std
        return data


class GrayImageChannelFormat(object):
    """
    format gray scale image's channel: (3,h,w) -> (1,h,w)
    Args:
        inverse: inverse gray image
    """

    def __init__(self, inverse=False, **kwargs):
        self.inverse = inverse

    def __call__(self, data):
        img = data["image"]
        img_single_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_expanded = np.expand_dims(img_single_channel, 0)

        if self.inverse:
            data["image"] = np.abs(img_expanded - 1)
        else:
            data["image"] = img_expanded

        data["src_image"] = img
        return data
