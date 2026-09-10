"""Run the two-stage margin segmentation pipeline on a folder of frames.

This is a thin wrapper around infer_sam2_red.py that fills in the validated
settings for the margin model, so there is nothing to remember:

  Stage 1  checkpoints_tumor_binary10/best_model_epoch_swa.pth
           Full-tumor segmentation. Its output is dilated by 20 px and used
           as a spatial constraint, so the margin can only be predicted on or
           near the tumor.

  Stage 2  checkpoints_margin58/best_model_epoch_swa.pth
           Margin segmentation. Predicts the uncharred region between the
           cutline and the tumor boundary, thresholded at 0.6 confidence.

EMA temporal smoothing (alpha 0.3) is applied to the margin confidence before
thresholding, which suppresses frame-to-frame flicker.

SAM 2 is deliberately disabled (--no-sam2). It drifts on this target.

Per-frame masks and overlays are written to the output folder, and the overlay
frames are then rendered into an .mp4 inside that same folder, named after the
input episode (e.g. full_balanced_exp1_20260602-001011-226627.mp4).

Usage:
    python run_margin.py <frames_dir> <output_dir>
    python run_margin.py <frames_dir> <output_dir> --threshold 0.5
    python run_margin.py <frames_dir> <output_dir> --fps 10
    python run_margin.py <frames_dir> <output_dir> --no-video

--fps (default 30) and --no-video are handled here. Any other argument is
passed through to infer_sam2_red.py, so a setting can be overridden without
editing this file.

The exact command this wrapper runs:

    python infer_sam2_red.py \
        --no-sam2 \
        --model checkpoints_margin58/best_model_epoch_swa.pth \
        --metadata checkpoints_margin58/best_model_epoch_swa_metadata.json \
        --input <frames_dir> \
        --output-dir <output_dir> \
        --threshold 0.6 \
        --blend-ema-alpha 0.3 \
        --spatial-model checkpoints_tumor_binary10/best_model_epoch_swa.pth \
        --spatial-metadata checkpoints_tumor_binary10/best_model_epoch_swa_metadata.json \
        --spatial-dilate 20 \
        --device cuda \
        --visualize
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Weight locations. Each can be overridden with an environment variable, so
# checkpoints kept outside the repository need no edits to this file.
MARGIN_MODEL = os.environ.get(
    "MARGIN_MODEL", os.path.join(HERE, "checkpoints_margin58", "best_model_epoch_swa.pth"))
MARGIN_META = os.environ.get(
    "MARGIN_META", os.path.join(HERE, "checkpoints_margin58", "best_model_epoch_swa_metadata.json"))
TUMOR_MODEL = os.environ.get(
    "SPATIAL_MODEL", os.path.join(HERE, "checkpoints_tumor_binary10", "best_model_epoch_swa.pth"))
TUMOR_META = os.environ.get(
    "SPATIAL_META", os.path.join(HERE, "checkpoints_tumor_binary10", "best_model_epoch_swa_metadata.json"))

FPS = 30            # frame rate of the rendered output video


def episode_name(frames_dir):
    """Name the episode from its input path, for use as the video filename.

    Frames live in <experiment>/<episode>/endoscope, so the identifying name
    is the two folders above the frames:

        G:/Ablation/full_balanced_exp1/20260602-001011-226627/endoscope
        -> full_balanced_exp1_20260602-001011-226627

    Generic container folder names ('endoscope', 'frames', 'images', 'rgb')
    are skipped. If the path is shallower than that, whatever folder names are
    available are used instead, so this never fails on an unexpected layout.
    """
    GENERIC = {"endoscope", "frames", "images", "img", "rgb", "png", "data"}
    parts = [p for p in os.path.normpath(os.path.abspath(frames_dir)).split(os.sep) if p]
    # Drop drive letters / root markers such as 'G:' so they never appear.
    parts = [p for p in parts if not p.endswith(":")]
    if not parts:
        return "output"
    if parts[-1].lower() in GENERIC:
        # Frames sit in a generic container, so the identifying name is the two
        # folders above it: <experiment>/<episode>/endoscope.
        parts = parts[:-1]
        name = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    else:
        # The folder already names the episode; use it as-is.
        name = parts[-1]
    # Keep the result usable as a filename on every platform.
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        raise SystemExit(2)

    frames_dir, out_dir = args[0], args[1]
    passthrough = args[2:]

    # --fps and --no-video belong to this wrapper, not to infer_sam2_red.py,
    # so pull them out before the rest is forwarded.
    fps = FPS
    make_video = True
    if "--no-video" in passthrough:
        make_video = False
        passthrough = [a for a in passthrough if a != "--no-video"]
    if "--fps" in passthrough:
        i = passthrough.index("--fps")
        if i + 1 >= len(passthrough):
            raise SystemExit("--fps requires a value")
        fps = passthrough[i + 1]
        passthrough = passthrough[:i] + passthrough[i + 2:]

    if not os.path.isdir(frames_dir):
        raise SystemExit(f"Input folder not found: {frames_dir}")

    missing = [p for p in (MARGIN_MODEL, MARGIN_META, TUMOR_MODEL, TUMOR_META)
               if not os.path.isfile(p)]
    if missing:
        print("Missing model weights:")
        for m in missing:
            print(f"  {m}")
        print("\nDownload them from the Google Drive link in the README and place")
        print("them in the checkpoints_margin58/ and checkpoints_tumor_binary10/ folders.")
        raise SystemExit(1)

    cmd = [
        sys.executable, os.path.join(HERE, "infer_sam2_red.py"),
        "--no-sam2",
        "--model", MARGIN_MODEL,
        "--metadata", MARGIN_META,
        "--input", frames_dir,
        "--output-dir", out_dir,
        "--threshold", "0.6",
        "--blend-ema-alpha", "0.3",
        "--spatial-model", TUMOR_MODEL,
        "--spatial-metadata", TUMOR_META,
        "--spatial-dilate", "20",
        "--device", "cuda",
        "--visualize",
    ]
    # argparse takes the LAST occurrence of a repeated flag, so anything the
    # caller passes here overrides the defaults set above.
    cmd += passthrough

    print("Running:")
    print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    print()
    rc = subprocess.call(cmd, cwd=HERE)
    if rc != 0:
        raise SystemExit(rc)

    if not make_video:
        raise SystemExit(0)

    # Render the overlay frames into a finished video, named after the input
    # episode so the file identifies its source.
    video_path = os.path.join(out_dir, episode_name(frames_dir) + ".mp4")
    print("\nRendering video...")
    vrc = subprocess.call([
        sys.executable, os.path.join(HERE, "frames_to_video.py"),
        "--input", out_dir,
        "--output", video_path,
        "--pattern", "*_overlay.png",
        "--fps", str(fps),
    ], cwd=HERE)
    if vrc != 0:
        print("Warning: video rendering failed, but the frames above are complete.")
        raise SystemExit(vrc)

    print(f"\nVideo: {video_path}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
