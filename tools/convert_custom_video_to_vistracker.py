#!/usr/bin/env python3
"""
Convert a custom single-view video package into the BEHAVE-style sequence layout
expected by the VisTracker demo pipeline.

This script is intentionally additive: it writes a converted sequence under
demo_data/ and avoids modifying the existing repo pipeline code.
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
import trimesh


TARGET_WIDTH = 2048
TARGET_HEIGHT = 1536
TARGET_KID = 1
DEFAULT_MAX_KID = 3
TARGET_FRAME_PREFIX = "t"
DEFAULT_GENDER = "male"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing custom video inputs.",
    )
    parser.add_argument(
        "--output-root",
        default="demo_data",
        help="Root directory where the converted sequence will be written.",
    )
    parser.add_argument(
        "--sequence-name",
        default=None,
        help="Optional output sequence name. Defaults to the input video stem without trailing camera id.",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=TARGET_WIDTH,
        help="Target padded image width.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=TARGET_HEIGHT,
        help="Target padded image height.",
    )
    parser.add_argument(
        "--kid",
        type=int,
        default=TARGET_KID,
        help="Kinect id to use for generated files.",
    )
    parser.add_argument(
        "--max-kid",
        type=int,
        default=DEFAULT_MAX_KID,
        help="Maximum kinect id to materialize compatibility intrinsics/config entries for.",
    )
    parser.add_argument(
        "--gender",
        default=DEFAULT_GENDER,
        help="Gender to place into info.json.",
    )
    parser.add_argument(
        "--object-name",
        default=None,
        help="Override object name. Defaults to parsing it from sequence name.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing output sequence before writing.",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def infer_sequence_name(input_dir: Path) -> str:
    video_files = sorted(input_dir.glob("*.color.mp4"))
    if len(video_files) != 1:
        raise ValueError(f"Expected exactly one *.color.mp4 in {input_dir}, found {len(video_files)}")
    stem = video_files[0].name
    suffix = ".color.mp4"
    if not stem.endswith(suffix):
        raise ValueError(f"Unexpected input video name: {video_files[0].name}")
    base = stem[: -len(suffix)]
    parts = base.split(".")
    if len(parts) > 1 and parts[-1].isdigit():
        return ".".join(parts[:-1])
    return base


def infer_object_name(seq_name: str) -> str:
    parts = seq_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot infer object name from sequence name: {seq_name}")
    return parts[2]


def extract_date_name(seq_name: str) -> str:
    parts = seq_name.split("_")
    if not parts:
        raise ValueError(f"Invalid sequence name: {seq_name}")
    return parts[0]


def center_pad_image(image: np.ndarray, target_width: int, target_height: int, pad_value=0):
    h, w = image.shape[:2]
    if w > target_width or h > target_height:
        raise ValueError(
            f"Input image {w}x{h} is larger than target canvas {target_width}x{target_height}"
        )
    dx = (target_width - w) // 2
    dy = (target_height - h) // 2
    if image.ndim == 2:
        canvas = np.full((target_height, target_width), pad_value, dtype=image.dtype)
        canvas[dy : dy + h, dx : dx + w] = image
    else:
        channels = image.shape[2]
        if np.isscalar(pad_value):
            pad_fill = np.full((target_height, target_width, channels), pad_value, dtype=image.dtype)
        else:
            pad_fill = np.zeros((target_height, target_width, channels), dtype=image.dtype)
            pad_fill[:] = np.asarray(pad_value, dtype=image.dtype)
        canvas = pad_fill
        canvas[dy : dy + h, dx : dx + w] = image
    return canvas, dx, dy


def threshold_mask(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return np.where(gray > 127, 255, 0).astype(np.uint8)


def make_body_json(kpts25_xyc: np.ndarray):
    body_joints = kpts25_xyc.reshape(-1).tolist()
    return {
        "body_joints": body_joints,
        "face_joints": [],
        "left_hand_joints": [],
        "right_hand_joints": [],
    }


def make_mocap_json(global_orient: np.ndarray, body_pose: np.ndarray, betas: np.ndarray, transl: np.ndarray):
    pose = np.concatenate([global_orient, body_pose], axis=0)
    return {
        "pose": pose.tolist(),
        "betas": betas[:10].tolist(),
        "trans": transl.tolist(),
        "bboxes": [0.0, 0.0, 0.0, 0.0],
        "is_pare": True,
        "orig_cam": [0.0, 0.0, 0.0, 0.0],
        "pred_cam": [0.0, 0.0, 0.0],
    }


def build_intrinsic_json(src_intrinsics, target_width, target_height, dx, dy):
    fx = float(src_intrinsics["fx"])
    fy = float(src_intrinsics["fy"])
    cx = float(src_intrinsics["cx"]) + dx
    cy = float(src_intrinsics["cy"]) + dy

    def make_camera(width, height, fx_, fy_, cx_, cy_):
        opencv = [fx_, fy_, cx_, cy_, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return {
            "codx": 0.0,
            "cody": 0.0,
            "cx": cx_,
            "cy": cy_,
            "fx": fx_,
            "fy": fy_,
            "height": height,
            "k1": 0.0,
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "k5": 0.0,
            "k6": 0.0,
            "metric_radius": 1.7,
            "opencv": opencv,
            "p1": 0.0,
            "p2": 0.0,
            "width": width,
        }

    depth_width = int(src_intrinsics.get("width", target_width))
    depth_height = int(src_intrinsics.get("height", target_height))
    depth_fx = fx
    depth_fy = fy
    depth_cx = float(src_intrinsics["cx"])
    depth_cy = float(src_intrinsics["cy"])

    return {
        "serial": "custom-single-view",
        "color": make_camera(target_width, target_height, fx, fy, cx, cy),
        "depth": make_camera(depth_width, depth_height, depth_fx, depth_fy, depth_cx, depth_cy),
        "color_to_depth": {
            "rotation": np.eye(3, dtype=np.float32).reshape(-1).tolist(),
            "translation": [0.0, 0.0, 0.0],
        },
        "depth_to_color": {
            "rotation": np.eye(3, dtype=np.float32).reshape(-1).tolist(),
            "translation": [0.0, 0.0, 0.0],
        },
    }


def build_pointcloud_table(depth_width: int, depth_height: int, fx: float, fy: float, cx: float, cy: float):
    xs, ys = np.meshgrid(np.arange(depth_width, dtype=np.float32), np.arange(depth_height, dtype=np.float32))
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    return np.dstack([x, y]).astype(np.float32)


def copy_intrinsics_bundle(intrinsic_root: Path, kid: int, src_intrinsics, target_width: int, target_height: int, dx: int, dy: int):
    dst = intrinsic_root / str(kid)
    ensure_dir(dst)

    calibration = build_intrinsic_json(src_intrinsics, target_width, target_height, dx, dy)
    save_json(dst / "calibration.json", calibration)

    fx = float(src_intrinsics["fx"])
    fy = float(src_intrinsics["fy"])
    cx = float(src_intrinsics["cx"])
    cy = float(src_intrinsics["cy"])
    pointcloud_table = build_pointcloud_table(
        int(src_intrinsics["width"]),
        int(src_intrinsics["height"]),
        fx,
        fy,
        cx,
        cy,
    )
    np.save(dst / "pointcloud_table.npy", pointcloud_table)

    calibs_payload = {
        "color": calibration["color"],
        "depth": calibration["depth"],
        "color_to_depth": calibration["color_to_depth"],
        "depth_to_color": calibration["depth_to_color"],
        "serial": calibration["serial"],
    }
    with open(dst / "calibs.pkl", "wb") as f:
        pickle.dump({"calib_raw": json.dumps({"custom_calibration": calibs_payload})}, f)

    npz_sidecar = dst / "calibs.pkl.npz"
    if npz_sidecar.exists():
        npz_sidecar.unlink()


def ensure_runtime_compat_intrinsics(intrinsic_root: Path, max_kid: int, src_intrinsics, target_width: int, target_height: int, dx: int, dy: int):
    for kid in range(max_kid + 1):
        copy_intrinsics_bundle(
            intrinsic_root=intrinsic_root,
            kid=kid,
            src_intrinsics=src_intrinsics,
            target_width=target_width,
            target_height=target_height,
            dx=dx,
            dy=dy,
        )


def simplify_mesh_to_face_count(mesh_path: Path, face_count: int):
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a mesh at {mesh_path}")

    if len(mesh.faces) <= face_count:
        return mesh

    try:
        import fast_simplification

        simple = fast_simplification.simplify_mesh(mesh, target_count=face_count)
        if isinstance(simple, trimesh.Trimesh):
            return simple
    except Exception:
        pass

    try:
        simple = mesh.simplify_quadratic_decimation(int(face_count))
        if isinstance(simple, trimesh.Trimesh) and len(simple.faces) > 0:
            return simple
    except Exception:
        pass

    try:
        import pymeshlab

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(mesh_path))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(face_count))
        current = ms.current_mesh()
        verts = current.vertex_matrix()
        faces = current.face_matrix()
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    except Exception:
        pass

    try:
        import open3d as o3d

        mesh_o3d = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
        if mesh_o3d.is_empty():
            raise RuntimeError("open3d loaded an empty mesh")
        simple = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=int(face_count))
        if simple.is_empty():
            raise RuntimeError("open3d simplification returned an empty mesh")
        verts = np.asarray(simple.vertices)
        faces = np.asarray(simple.triangles)
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to simplify mesh {mesh_path} to {face_count} faces: {exc}")


def export_object_assets(src_obj_dir: Path, dst_obj_dir: Path, object_name: str, face_count: int):
    ensure_dir(dst_obj_dir)

    src_obj = src_obj_dir / f"{object_name}.obj"
    if not src_obj.exists():
        src_obj = src_obj_dir / "box.obj"
    if not src_obj.exists():
        candidates = sorted(src_obj_dir.glob("*.obj"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Cannot find a unique source OBJ in {src_obj_dir}")
        src_obj = candidates[0]

    mesh = simplify_mesh_to_face_count(src_obj, face_count)
    mesh.export(dst_obj_dir / f"{object_name}_f{face_count}.ply")

    copy_exts = [".obj", ".mtl", ".png", ".jpg", ".jpeg", ".texture.png"]
    for ext in copy_exts:
        for file in src_obj_dir.glob(f"*{ext}"):
            shutil.copy2(file, dst_obj_dir / file.name)

    exported_obj = dst_obj_dir / f"{object_name}.obj"
    if not exported_obj.exists():
        shutil.copy2(src_obj, exported_obj)
    return dst_obj_dir / f"{object_name}_f{face_count}.ply"


def frame_name_from_index(index: int, fps: float) -> str:
    timestamp = index / fps
    return f"{TARGET_FRAME_PREFIX}{timestamp:08.3f}"


def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def validate_lengths(video_caps, expected_frames):
    counts = []
    for cap in video_caps:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        counts.append(count)
    if len(set(counts)) != 1:
        raise ValueError(f"Input videos have inconsistent frame counts: {counts}")
    if expected_frames is not None and counts[0] != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, found {counts[0]}")
    return counts[0]


def convert_sequence(args):
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    seq_name = args.sequence_name or infer_sequence_name(input_dir)
    object_name = args.object_name or infer_object_name(seq_name)
    date_name = extract_date_name(seq_name)

    video_path = next(input_dir.glob("*.color.mp4"))
    human_mask_path = input_dir / "mask_human.mp4"
    object_mask_path = input_dir / "mask_object.mp4"
    keypoint_path = input_dir / "keypoint25.npz"
    smpl_path = input_dir / "smpl_params_init.npz"
    intrinsics_path = input_dir / "camera_intrinsics.json"
    object_mesh_dir = input_dir / object_name

    required = [
        human_mask_path,
        object_mask_path,
        keypoint_path,
        smpl_path,
        intrinsics_path,
        object_mesh_dir,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    output_root = Path(args.output_root).resolve()
    seq_dir = output_root / seq_name
    if seq_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output sequence already exists: {seq_dir}. Use --force to overwrite.")
        shutil.rmtree(seq_dir)
    ensure_dir(seq_dir)

    calibs_root = output_root / "calibs"
    shared_config_root = calibs_root / date_name / "config"
    intrinsic_root = seq_dir / "intrinsics"
    template_root = seq_dir / "object_templates" / object_name
    ensure_dir(intrinsic_root)

    if not shared_config_root.exists():
        raise FileNotFoundError(
            f"Expected shared BEHAVE config to exist at {shared_config_root}"
        )

    intrinsics = load_json(intrinsics_path)
    src_width = int(intrinsics["width"])
    src_height = int(intrinsics["height"])
    if src_width > args.target_width or src_height > args.target_height:
        raise ValueError(
            f"Input size {src_width}x{src_height} exceeds target {args.target_width}x{args.target_height}"
        )
    dx = (args.target_width - src_width) // 2
    dy = (args.target_height - src_height) // 2

    ensure_runtime_compat_intrinsics(
        intrinsic_root=intrinsic_root,
        max_kid=max(args.kid, args.max_kid),
        src_intrinsics=intrinsics,
        target_width=args.target_width,
        target_height=args.target_height,
        dx=dx,
        dy=dy,
    )
    template_mesh = export_object_assets(object_mesh_dir, template_root, object_name, face_count=2000)

    info = {
        "config": f"../calibs/{date_name}/config",
        "intrinsic": "intrinsics",
        "cat": object_name,
        "gender": args.gender,
        "empty": None,
        "kinects": list(range(max(args.kid, args.max_kid) + 1)),
        "beta": None,
        "template_dir": str(template_root.relative_to(seq_dir)),
        "template": str(template_mesh.relative_to(seq_dir)),
    }
    save_json(seq_dir / "info.json", info)

    kpt_npz = np.load(keypoint_path, allow_pickle=True)
    smpl_npz = np.load(smpl_path, allow_pickle=True)

    if "keypoint25_xyc" not in kpt_npz:
        raise KeyError(f"keypoint25_xyc not found in {keypoint_path}")
    kpts = np.asarray(kpt_npz["keypoint25_xyc"], dtype=np.float32)
    kpts = kpts.copy()
    kpts[..., 0] += dx
    kpts[..., 1] += dy

    global_orient = np.asarray(smpl_npz["global_orient"], dtype=np.float32)
    body_pose = np.asarray(smpl_npz["body_pose"], dtype=np.float32)
    betas = np.asarray(smpl_npz["betas"], dtype=np.float32)
    transl = np.asarray(smpl_npz["transl"], dtype=np.float32)

    caps = [
        open_video(video_path),
        open_video(human_mask_path),
        open_video(object_mask_path),
    ]
    try:
        frame_count = validate_lengths(caps, expected_frames=kpts.shape[0])
        if global_orient.shape[0] != frame_count or body_pose.shape[0] != frame_count or betas.shape[0] != frame_count or transl.shape[0] != frame_count:
            raise ValueError("SMPL init arrays do not match video frame count")

        fps = caps[0].get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        for frame_idx in range(frame_count):
            ok_rgb, rgb = caps[0].read()
            ok_h, human_mask_bgr = caps[1].read()
            ok_o, object_mask_bgr = caps[2].read()
            if not (ok_rgb and ok_h and ok_o):
                raise RuntimeError(f"Failed to read synchronized frame {frame_idx}")

            rgb_pad, _, _ = center_pad_image(rgb, args.target_width, args.target_height, pad_value=0)
            human_mask = threshold_mask(human_mask_bgr)
            human_pad, _, _ = center_pad_image(human_mask, args.target_width, args.target_height, pad_value=0)
            object_mask = threshold_mask(object_mask_bgr)
            object_pad, _, _ = center_pad_image(object_mask, args.target_width, args.target_height, pad_value=0)

            frame_dir = seq_dir / frame_name_from_index(frame_idx, fps)
            ensure_dir(frame_dir)

            cv2.imwrite(str(frame_dir / f"k{args.kid}.color.jpg"), rgb_pad)
            cv2.imwrite(str(frame_dir / f"k{args.kid}.person_mask.png"), human_pad)
            cv2.imwrite(str(frame_dir / f"k{args.kid}.obj_rend_mask.png"), object_pad)

            color_json = make_body_json(kpts[frame_idx])
            save_json(frame_dir / f"k{args.kid}.color.json", color_json)

            mocap_json = make_mocap_json(
                global_orient=global_orient[frame_idx],
                body_pose=body_pose[frame_idx],
                betas=betas[frame_idx],
                transl=transl[frame_idx],
            )
            save_json(frame_dir / f"k{args.kid}.mocap.json", mocap_json)
    finally:
        for cap in caps:
            cap.release()

    summary = {
        "input_dir": str(input_dir),
        "sequence_name": seq_name,
        "object_name": object_name,
        "frames": int(frame_count),
        "fps": float(fps),
        "source_size": [src_width, src_height],
        "target_size": [args.target_width, args.target_height],
        "pad_offset_xy": [dx, dy],
        "seq_dir": str(seq_dir),
        "kid": args.kid,
    }
    save_json(seq_dir / "conversion_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    convert_sequence(args)


if __name__ == "__main__":
    main()
