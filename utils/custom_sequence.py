import json
import os.path as osp
from glob import glob

from behave.seq_utils import SeqInfo


def use_custom_video(args):
    return bool(getattr(args, "custom_video", False))


def load_info_json(seq_folder):
    info_file = osp.join(seq_folder, "info.json")
    with open(info_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_color_intrinsics(seq_folder, kid=1):
    seq_info = SeqInfo(seq_folder)
    calib_file = osp.join(seq_info.get_intrinsic(), str(kid), "calibration.json")
    with open(calib_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["color"]


def get_pixel_camera_params(seq_folder, kid=1):
    color = _load_color_intrinsics(seq_folder, kid)
    return {
        "fx": float(color["fx"]),
        "fy": float(color["fy"]),
        "cx": float(color["cx"]),
        "cy": float(color["cy"]),
        "image_width": int(color["width"]),
        "image_height": int(color["height"]),
    }


def get_normalized_camera_params(seq_folder, kid=1, crop_size=1200):
    params = get_pixel_camera_params(seq_folder, kid)
    image_width = float(params["image_width"])
    return {
        "crop_size": int(crop_size),
        "fx": params["fx"] / image_width,
        "fy": params["fy"] / image_width,
        "cx": params["cx"] / image_width,
        "cy": params["cy"] / image_width,
        "image_width": params["image_width"],
        "image_height": params["image_height"],
    }


def get_custom_template_path(seq_folder, obj_name=None):
    info = load_info_json(seq_folder)
    template_rel = info.get("template")
    if template_rel:
        template_path = osp.join(seq_folder, template_rel)
        if osp.isfile(template_path):
            return template_path
    template_dir = info.get("template_dir")
    if template_dir:
        template_dir = osp.join(seq_folder, template_dir)
        candidates = sorted(glob(osp.join(template_dir, "*.ply")))
        if obj_name is not None:
            obj_hits = [x for x in candidates if osp.basename(x).startswith(f"{obj_name}_")]
            if len(obj_hits) > 0:
                return obj_hits[0]
        if len(candidates) > 0:
            return candidates[0]
    raise FileNotFoundError(f"Cannot find custom object template in {seq_folder}")
