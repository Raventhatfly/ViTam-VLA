#!/usr/bin/env python3
"""
Use MovieChat to ask questions about a LIBERO HDF5 demo.

Two modes (mirrors MovieChat's official inference.py logic)
────────────────────────────────────────────────────────────
Global mode (default) — ask about the whole clip:
    python test_scripts/moviechat_qa.py \
        --hdf5 demo.hdf5 --demo demo_0 \
        --question "What is the robot doing?"

Breakpoint mode (--frame N) — ask about a specific moment:
    python test_scripts/moviechat_qa.py \
        --hdf5 demo.hdf5 --demo demo_0 \
        --frame 80 \
        --question "Has the robot grasped the object yet?"

Works for plain video files too (--video instead of --hdf5).
For video files, --frame is in seconds (float).

Other flags:
    --camera      HDF5 camera key (default: agentview_rgb)
    --n_samples   Number of fragments the video is split into (default: 128,
                  same as MovieChat's N_SAMPLES)
    --frms_per_frag  Frames loaded per fragment (default: 8, MAX_INT in source)
    --llama_model Path or HF repo ID for Vicuna LLM
    --ckpt        Local .pth checkpoint; omit to auto-download via from_config()
    --device      e.g. cuda:0

Environment:
    conda activate memory
"""

import argparse
import sys
import os
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import cv2


# ─── model loading ────────────────────────────────────────────────────────────

def build_model_from_ckpt(llama_model: str, ckpt_path: str, device):
    from MovieChat.models.moviechat import MovieChat
    print("Building MovieChat model…")
    model = MovieChat(
        vit_model="eva_clip_g",
        q_former_model=(
            "https://storage.googleapis.com/sfr-vision-language-research/"
            "LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
        ),
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model=llama_model,
        llama_proj_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=160,
        end_sym="###",
        low_resource=False,
        frozen_llama_proj=True,
        frozen_video_Qformer=True,
        fusion_header_type="seqTransf",
        max_frame_pos=32,
        fusion_head_layers=2,
        num_video_query_token=32,
    )
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"  missing: {len(msg.missing_keys)}  unexpected: {len(msg.unexpected_keys)}")
    return model


def build_model_auto(device):
    from MovieChat.models.moviechat import MovieChat
    print("Building MovieChat model via from_config (may download weights)…")
    return MovieChat.from_config(device)


# ─── HDF5 helpers ─────────────────────────────────────────────────────────────

def load_hdf5_frames(hdf5_path, demo, camera):
    """Return all frames as (T, H, W, 3) uint8 and validate keys."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        demos = list(f["data"].keys())
        if demo not in demos:
            sys.exit(f"[error] demo '{demo}' not found.\nAvailable: {sorted(demos)}")
        cameras = list(f[f"data/{demo}/obs"].keys())
        if camera not in cameras:
            sys.exit(f"[error] camera '{camera}' not found.\nAvailable: {cameras}")
        frames = f[f"data/{demo}/obs/{camera}"][()]   # (T, H, W, C)

    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    return frames  # (T, H, W, 3) RGB


def frames_to_tensor(frames_np):
    """(T, H, W, 3) uint8  →  (C, T, H, W) float32, resized to 224."""
    t = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()  # (C, T, H, W)
    if t.shape[2] != 224 or t.shape[3] != 224:
        t = F.interpolate(
            t.permute(1, 0, 2, 3),           # (T, C, H, W)
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3)
    return t  # (C, T, H, W)


def single_frame_to_tensor(frame_np, image_vis_processor, device):
    """
    Encode one (H, W, 3) uint8 frame the same way MovieChat's inference.py does:
        cv2 → save jpg → PIL → image_vis_processor → encode_image
    Returns cur_image: (1, num_query, hidden).
    """
    from PIL import Image

    # write to temp jpg so PIL can open it (matches official flow exactly)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
    raw_image = Image.open(tmp.name).convert("RGB")
    os.unlink(tmp.name)

    # [1, 3, 1, 224, 224]
    image_tensor = image_vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(device)
    return image_tensor   # caller will pass to model.encode_image()


# ─── core inference ───────────────────────────────────────────────────────────

def run_moviechat(model, vis_processor, frames_all, cur_frame_np,
                  middle_video, frame_idx, n_samples, frms_per_frag, device):
    """
    Replicates inference.py's upload_video_without_audio + encode flow.

    frames_all   : (T, H, W, 3) uint8 — full clip (or up to moment)
    cur_frame_np : (H, W, 3) uint8    — the exact moment frame
    middle_video : bool
    frame_idx    : int — position of the moment within frames_all (for cur_frame calc)
    """
    from MovieChat.processors.blip_processors import Blip2ImageEvalProcessor

    image_vis_processor = Blip2ImageEvalProcessor()

    # ── encode cur_image (always, both modes) ────────────────────────────────
    image_tensor = single_frame_to_tensor(cur_frame_np, image_vis_processor, device)
    cur_image = model.encode_image(image_tensor)   # (1, q, h)

    # ── reset memory buffers ─────────────────────────────────────────────────
    model.short_memory_buffer = []
    model.long_memory_buffer  = []
    model.temp_short_memory   = []

    T = len(frames_all)

    # ── split into n_samples fragments and encode each ───────────────────────
    # Official: N_SAMPLES=128 fragments, MAX_INT=8 frames per fragment
    # num_frames = how many fragments to process
    per_frag_len = max(1, T // n_samples)

    if middle_video:
        num_fragments = frame_idx // per_frag_len  if per_frag_len > 0 else 0
        # cur_frame = position of moment within the last fragment
        per_frame_step = max(1, per_frag_len // frms_per_frag)
        cur_frame_in_frag = (frame_idx - per_frag_len * num_fragments) // per_frame_step
    else:
        num_fragments = T // per_frag_len
        cur_frame_in_frag = 0

    if num_fragments == 0:
        # video too short — encode everything as one fragment
        frag_np = frames_all
        frag_t  = frames_to_tensor(frag_np)
        frag_t  = vis_processor.transform(frag_t).unsqueeze(0).to(device)
        model.encode_short_memory_frame(frag_t, cur_frame_in_frag)
    else:
        for i in range(num_fragments):
            start = i * per_frag_len
            end   = min(start + per_frag_len, T)
            frag_np = frames_all[start:end]

            # sample frms_per_frag frames from this fragment
            indices = np.linspace(0, len(frag_np) - 1, min(frms_per_frag, len(frag_np)), dtype=int)
            frag_np = frag_np[indices]

            frag_t = frames_to_tensor(frag_np)
            frag_t = vis_processor.transform(frag_t).unsqueeze(0).to(device)

            if middle_video and (i + 1) == num_fragments:
                model.encode_short_memory_frame(frag_t, cur_frame_in_frag)
            else:
                model.encode_short_memory_frame(frag_t)

    # ── aggregate into video embedding ───────────────────────────────────────
    video_emb, _ = model.encode_long_video(cur_image, middle_video)

    fps_approx = 20.0
    total_used = num_fragments * per_frag_len
    msg = (f"The video contains {total_used} frames "
           f"({total_used / fps_approx:.1f} seconds). ")
    return [video_emb], msg


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ask MovieChat about a whole video or a specific moment."
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Path to a video file (.mp4, .avi, …)")
    src.add_argument("--hdf5",  help="Path to a LIBERO HDF5 demonstration file")

    parser.add_argument("--demo",   default="demo_0")
    parser.add_argument("--camera", default="agentview_rgb")
    parser.add_argument("--frame",  type=int, default=None,
                        help="HDF5 frame index of the moment (breakpoint mode). "
                             "Omit for global mode.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--device",   default="cuda:0")
    parser.add_argument("--n_samples",     type=int, default=128,
                        help="Number of fragments to split video into (default: 128)")
    parser.add_argument("--frms_per_frag", type=int, default=8,
                        help="Frames sampled per fragment (default: 8)")
    parser.add_argument("--llama_model", default="Enxin/MovieChat-vicuna")
    parser.add_argument("--ckpt",        default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── load model ───────────────────────────────────────────────────────────
    if args.ckpt:
        model = build_model_from_ckpt(args.llama_model, args.ckpt, device)
    else:
        model = build_model_auto(device)
    model = model.to(device).eval()

    from MovieChat.processors.video_processor import AlproVideoEvalProcessor
    vis_processor = AlproVideoEvalProcessor(image_size=224, n_frms=args.frms_per_frag)

    # ── load frames ──────────────────────────────────────────────────────────
    if args.hdf5:
        if not os.path.isfile(args.hdf5):
            sys.exit(f"[error] HDF5 not found: {args.hdf5}")
        frames_all = load_hdf5_frames(args.hdf5, args.demo, args.camera)
    else:
        sys.exit("[error] --video support not implemented yet; use --hdf5")

    T = len(frames_all)
    middle_video = args.frame is not None
    frame_idx    = args.frame if middle_video else 0   # 0 = first frame for global

    frame_idx = max(0, min(frame_idx, T - 1))
    cur_frame_np = frames_all[frame_idx]   # (H, W, 3) the snapshot frame

    mode_str = f"breakpoint (frame {frame_idx}/{T})" if middle_video else "global"
    print(f"Mode: {mode_str}  |  clip length: {T} frames")

    # ── encode & answer ──────────────────────────────────────────────────────
    from MovieChat.models.chat_model import Chat
    chat = Chat(model, vis_processor, device)

    img_list, msg = run_moviechat(
        model, vis_processor,
        frames_all, cur_frame_np,
        middle_video, frame_idx,
        args.n_samples, args.frms_per_frag, device,
    )

    print(f"\nQ: {args.question}")
    answer, _ = chat.answer(img_list=img_list, input_text=args.question, msg=msg)
    print(f"A: {answer}")


if __name__ == "__main__":
    main()
