#!/usr/bin/env python3
"""
Export a LIBERO HDF5 demonstration to an MP4 video.

Usage:
    python test_scripts/hdf5_to_mp4.py \
        --hdf5 /path/to/task_demo.hdf5 \
        --demo demo_0 \
        --camera agentview_rgb \
        --out /tmp/demo_0.mp4

    # export all demos in one file, side-by-side with two cameras:
    python test_scripts/hdf5_to_mp4.py \
        --hdf5 /path/to/task_demo.hdf5 \
        --all \
        --camera agentview_rgb eye_in_hand_rgb \
        --outdir /tmp/demos/
"""

import argparse
import os
import sys
import h5py
import numpy as np
import cv2


def list_cameras(f, demo):
    return list(f[f"data/{demo}/obs"].keys())


def get_frames(f, demo, camera, vflip=False):
    """Returns (T, H, W, 3) uint8 array."""
    arr = f[f"data/{demo}/obs/{camera}"][()]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    # drop alpha channel if present
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    # Some renderers store image origin at bottom-left (OpenGL style).
    if vflip:
        arr = arr[:, ::-1, :, :]
    return arr


def write_mp4(frames, out_path, fps=20):
    """frames: (T, H, W, 3) uint8 RGB"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    T, H, W, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  saved → {out_path}  ({T} frames @ {fps} fps)")


def make_side_by_side(arrays):
    """Horizontally concatenate multiple (T, H, W, 3) arrays (resize to same H)."""
    if len(arrays) == 1:
        return arrays[0]
    T = arrays[0].shape[0]
    H = arrays[0].shape[1]
    resized = []
    for arr in arrays:
        if arr.shape[0] != T:
            # trim/pad to T
            arr = arr[:T] if arr.shape[0] >= T else np.pad(
                arr, ((0, T - arr.shape[0]), (0, 0), (0, 0), (0, 0)), mode="edge"
            )
        if arr.shape[1] != H:
            # resize height to match first camera
            new_frames = []
            for f in arr:
                new_frames.append(cv2.resize(f, (int(f.shape[1] * H / f.shape[0]), H)))
            arr = np.stack(new_frames)
        resized.append(arr)
    return np.concatenate(resized, axis=2)  # concat along width


def export_demo(f, demo, cameras, out_path, fps, vflip=False):
    arrays = [get_frames(f, demo, cam, vflip=vflip) for cam in cameras]
    combined = make_side_by_side(arrays)
    write_mp4(combined, out_path, fps)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a LIBERO HDF5 demo to MP4."
    )
    parser.add_argument("--hdf5", required=True, help="Path to HDF5 file")
    parser.add_argument("--demo", default="demo_0",
                        help="Demo key to export (default: demo_0). Ignored if --all.")
    parser.add_argument("--all", action="store_true",
                        help="Export every demo in the file")
    parser.add_argument("--camera", nargs="+", default=["agentview_rgb"],
                        help="Camera key(s) to include. Multiple cameras are placed side-by-side.")
    parser.add_argument("--out", default=None,
                        help="Output MP4 path (single-demo mode)")
    parser.add_argument("--outdir", default=".",
                        help="Output directory (--all mode, default: current dir)")
    parser.add_argument("--fps", type=int, default=20, help="Frame rate (default: 20)")
    parser.add_argument("--vflip", action="store_true",
                        help="Vertically flip frames before export (fix upside-down videos)")
    parser.add_argument("--list", action="store_true",
                        help="Just print available demos and cameras, then exit")
    args = parser.parse_args()

    if not os.path.isfile(args.hdf5):
        sys.exit(f"[error] file not found: {args.hdf5}")

    with h5py.File(args.hdf5, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1]))

        if args.list:
            print(f"Demos ({len(demos)}): {demos}")
            print(f"Cameras: {list_cameras(f, demos[0])}")
            return

        # validate cameras
        available = list_cameras(f, demos[0])
        for cam in args.camera:
            if cam not in available:
                sys.exit(f"[error] camera '{cam}' not found. Available: {available}")

        if args.all:
            os.makedirs(args.outdir, exist_ok=True)
            for demo in demos:
                stem = os.path.splitext(os.path.basename(args.hdf5))[0]
                out_path = os.path.join(args.outdir, f"{stem}_{demo}.mp4")
                print(f"Exporting {demo}…")
                export_demo(f, demo, args.camera, out_path, args.fps, vflip=args.vflip)
        else:
            if args.demo not in demos:
                sys.exit(
                    f"[error] demo '{args.demo}' not found.\n"
                    f"Available: {demos}"
                )
            if args.out is None:
                stem = os.path.splitext(os.path.basename(args.hdf5))[0]
                args.out = f"{stem}_{args.demo}.mp4"
            print(f"Exporting {args.demo}…")
            export_demo(f, args.demo, args.camera, args.out, args.fps, vflip=args.vflip)


if __name__ == "__main__":
    main()
