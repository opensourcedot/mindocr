import glob
import logging
import os

import numpy as np
from shapely.geometry import Polygon

_logger = logging.getLogger("mindocr")


def get_ckpt_file(ckpt_dir):
    if os.path.isfile(ckpt_dir):
        ckpt_load_path = ckpt_dir
    else:
        ckpt_paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
        assert len(ckpt_paths) != 0, f"No .ckpt files found in {ckpt_dir}"
        ckpt_load_path = ckpt_paths[0]
        if len(ckpt_paths) > 1:
            _logger.warning(f"More than one .ckpt files found in {ckpt_dir}. Pick {ckpt_load_path}")

    return ckpt_load_path


def validate_det_res(det_res, img_shape, order_clockwise=True, min_poly_points=3, min_area=3):
    polys = det_res["polys"].copy()
    scores = det_res.get("scores", [])

    if len(polys) == 0:
        return dict(polys=[], scores=[])

    h, w = img_shape[:2]
    # clip if ouf of image
    if not isinstance(polys, list):
        polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
        polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)
    else:
        for i, poly in enumerate(polys):
            polys[i][:, 0] = np.clip(polys[i][:, 0], 0, w - 1)
            polys[i][:, 1] = np.clip(polys[i][:, 1], 0, h - 1)

    new_polys = []
    if scores is not None:
        new_scores = []
    for i, poly in enumerate(polys):
        # refine points to clockwise order
        if order_clockwise:
            if len(poly) == 4:
                poly = order_points_clockwise(poly)
            else:
                _logger.warning("order_clockwise only supports quadril polygons currently")
        # filter
        if len(poly) < min_poly_points:
            continue

        if min_area > 0:
            p = Polygon(poly)
            if p.is_valid and not p.is_empty:
                if p.area >= min_area:
                    poly_np = np.array(p.exterior.coords)[:-1, :]
                    new_polys.append(poly_np)
                    if scores is not None:
                        new_scores.append(scores[i])
        else:
            new_polys.append(poly)
            if scores is not None:
                new_scores.append(scores[i])

    if len(scores) > 0:
        new_det_res = dict(polys=np.array(new_polys, dtype=int), scores=new_scores)
    else:
        new_det_res = dict(polys=np.array(new_polys, dtype=int))

    return new_det_res


def order_points_clockwise(points):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    tmp = np.delete(points, (np.argmin(s), np.argmax(s)), axis=0)
    diff = np.diff(np.array(tmp), axis=1)
    rect[1] = tmp[np.argmin(diff)]
    rect[3] = tmp[np.argmax(diff)]

    return rect
