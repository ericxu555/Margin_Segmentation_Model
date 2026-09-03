#!/usr/bin/env python3
"""
General VIDEO inference: margin (red) OR binary tumor, SAM 2 tracking OPTIONAL.

NOTE ON NAME: Named for its original SAM2+red role, but this is now the general
video-inference script. It handles BOTH the margin (red) model AND binary tumor
tracking. SAM 2 is OPTIONAL — pass --no-sam2 for per-frame inference (this is the
DEFAULT mode for margin models; see the no-SAM2-for-margin rule). The name is
kept for continuity. For simple binary tumor per-frame inference, prefer
infer_epoch61.py.

--- Original description (SAM 2 hybrid mode) ---
SAM 2 + Red Binary Model hybrid inference for remaining tumor segmentation.

Pipeline:
  1. Run the red binary UNet on frame 1 to obtain the initial red mask.
  2. Use that mask as SAM 2 object prompt (obj_id=1, remaining tumor).
  3. Propagate SAM 2 tracking forward through all remaining frames.
  4. Optionally blend the red binary model every frame to correct SAM 2 drift.

Blue (resected tumor) is not tracked — the red binary model only predicts
remaining tumor vs background.

Usage example::

    python infer_sam2_red.py \\
        --model checkpoints_margin58/best_model_epoch_swa.pth \\
        --metadata checkpoints_margin58/best_model_epoch_swa_metadata.json \\
        --input "path/to/frames" \\
        --output-dir "path/to/output" \\
        --visualize \\
        --blend-epoch61 --blend-confidence 0.5 --blend-ema-alpha 0.7 \\
        --blend-carry-frames 5 --carry-anchor --carry-anchor-high 0.5
"""

import argparse
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from src.inference_engine import InferenceEngine
from src.visualization import VisualizationGenerator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT_NAME = "sam2.1_hiera_small.pt"

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_sam2_checkpoint(checkpoint_dir: Path) -> Path:
    """Download SAM 2.1 hiera_small checkpoint if not already present."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / SAM2_CHECKPOINT_NAME
    if not checkpoint_path.exists():
        print(f"Downloading SAM 2.1 checkpoint to {checkpoint_path} ...")

        def _progress(count, block_size, total_size):
            if total_size > 0:
                pct = min(count * block_size / total_size * 100, 100)
                print(f"\r  {pct:5.1f}%", end="", flush=True)

        urllib.request.urlretrieve(SAM2_CHECKPOINT_URL, checkpoint_path, _progress)
        print()
        print("  Download complete.")
    else:
        print(f"SAM 2 checkpoint found at {checkpoint_path}")
    return checkpoint_path


def collect_frames(input_dir: Path) -> List[Path]:
    """Return sorted frame paths, excluding mask/org/overlay outputs."""
    return sorted([
        p for p in input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.stem.endswith('_mask')
        and not p.stem.endswith('_org')
        and not p.stem.endswith('_overlay')
    ])


def compose_rgb_mask(
    remaining: np.ndarray,
    h: int,
    w: int,
    green: Optional[np.ndarray] = None,
    color: tuple = (255, 0, 0),
) -> np.ndarray:
    """Compose primary (and optional green) binary masks into a single RGB mask."""
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[remaining] = color
    if green is not None and green.any():
        rgb[green] = [0, 255, 0]
    return rgb


def _rotate_img_tta(img: np.ndarray, angle: float) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def _rotate_prob_tta(prob: np.ndarray, angle: float) -> np.ndarray:
    import cv2
    h, w = prob.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(prob, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def predict_red_512(engine, img: np.ndarray, tta: bool) -> np.ndarray:
    """Return 512×512 red-class probability as float32 ndarray (TTA or single-pass)."""
    if not tta:
        probs, _ = engine.predict_single(img)
        return probs[1].float().cpu().numpy()
    angles = [0, 10, -10, 15, -15]
    red_probs = []
    for angle in angles:
        aug = _rotate_img_tta(img, angle) if angle != 0 else img
        probs, _ = engine.predict_single(aug)
        rp = probs[1].float().cpu().numpy()
        if angle != 0:
            rp = _rotate_prob_tta(rp, -angle)
        red_probs.append(rp)
    return np.stack(red_probs).mean(axis=0)


def load_precomputed_spatial_masks(mask_dir: Path, frame_paths: list) -> dict:
    """
    Build a stem→mask-path lookup from a pre-computed spatial mask directory.
    Matches each input frame stem (e.g. 'frame_0001') to its corresponding
    '_mask.png' file (e.g. 'frame_0001_mask.png').
    """
    lookup = {}
    for mp in mask_dir.iterdir():
        if mp.is_file() and mp.suffix.lower() in IMAGE_EXTENSIONS and '_mask' in mp.stem:
            # stem without '_mask' suffix → original frame stem
            frame_stem = mp.stem.replace('_mask', '')
            lookup[frame_stem] = mp
    matched = sum(1 for fp in frame_paths if fp.stem in lookup)
    print(f"Spatial mask dir: {matched}/{len(frame_paths)} frames matched.")
    return lookup


def get_precomputed_spatial_mask(
    lookup: dict,
    frame_path: Path,
    h_orig: int,
    w_orig: int,
    dilate_px: int,
) -> np.ndarray:
    """Load and binarise a pre-computed spatial mask, resize to original resolution, dilate."""
    import cv2
    mp = lookup.get(frame_path.stem)
    if mp is None:
        return np.ones((h_orig, w_orig), dtype=bool)   # no mask → allow everywhere
    arr = np.array(Image.open(mp).convert('RGB').resize((w_orig, h_orig), Image.NEAREST))
    tumor = (arr[:, :, 0] > 0) | (arr[:, :, 1] > 0) | (arr[:, :, 2] > 0)
    tumor_u8 = tumor.astype(np.uint8)
    if dilate_px > 0:
        k = max(dilate_px | 1, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tumor_u8 = cv2.dilate(tumor_u8, kernel)
    return tumor_u8.astype(bool)


def get_spatial_mask(
    spatial_engine,
    img: np.ndarray,
    h_orig: int,
    w_orig: int,
    dilate_px: int,
) -> np.ndarray:
    """
    Run the spatial constraint model (epoch 60 4-class) and return a boolean mask
    of the tumor region (any non-background class), dilated by dilate_px pixels.
    The dilation ensures the red margin at the tumor boundary is not clipped.
    """
    import cv2
    probs, _ = spatial_engine.predict_single(img)          # (C, 512, 512)
    tumor_512 = (probs.argmax(dim=0).cpu().numpy() > 0).astype(np.uint8)
    tumor_orig = np.array(
        Image.fromarray(tumor_512).resize((w_orig, h_orig), Image.NEAREST)
    )
    if dilate_px > 0:
        k = max(dilate_px | 1, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tumor_orig = cv2.dilate(tumor_orig, kernel)
    return tumor_orig.astype(bool)


def run_model_fallback(
    frame_paths: List[Path],
    engine: InferenceEngine,
    output_dir: Path,
    h_orig: int,
    w_orig: int,
    viz_generator: Optional[VisualizationGenerator],
    alpha: float,
) -> float:
    """Per-frame red binary model inference (no SAM 2). Returns elapsed seconds."""
    from src.postprocessor import MaskPostProcessor
    postprocessor = MaskPostProcessor(original_size=(h_orig, w_orig))

    t0 = time.perf_counter()
    for i, fp in enumerate(frame_paths, 1):
        img = np.array(Image.open(fp).convert('RGB'))
        probs, _ = engine.predict_single(img)           # (2, 512, 512)
        rgb_mask = postprocessor.postprocess_mask(probs.unsqueeze(0))

        stem = fp.stem
        Image.fromarray(rgb_mask.astype(np.uint8)).save(output_dir / f'{stem}_mask.png')
        Image.open(fp).save(output_dir / f'{stem}_org.png')

        if viz_generator is not None:
            overlay = viz_generator.generate_overlay(img, rgb_mask, alpha=alpha)
            Image.fromarray(overlay.astype(np.uint8)).save(
                output_dir / f'{stem}_overlay.png'
            )

        print(f"  [{i}/{len(frame_paths)}] {fp.name}")

    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='SAM 2 + Red Binary Model hybrid tumor segmentation inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model',
        type=str,
        default='checkpoints_margin58/best_model_epoch_swa.pth',
        help='Path to red binary UNet checkpoint (.pth)',
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default='checkpoints_margin58/best_model_epoch_swa_metadata.json',
        help='Path to red binary model metadata (.json)',
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Directory of input frames',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs_sam2_red',
        help='Output directory for masks, originals, and overlays',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
    )
    parser.add_argument(
        '--sam2-device',
        type=str,
        default='cpu',
        choices=['cuda', 'cpu'],
        help=(
            'Device for SAM 2 video predictor. Defaults to cpu because SAM 2 '
            'triggers CUDA context corruption on some Windows + Blackwell GPU '
            'configurations. Use --sam2-device cuda if your system handles it.'
        ),
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Save colour overlay images ({stem}_overlay.png)',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Overlay transparency (0 = opaque mask, 1 = original only)',
    )
    parser.add_argument(
        '--sam2-checkpoint-dir',
        type=str,
        default='checkpoints_sam2',
        help='Directory to store / load the SAM 2 checkpoint',
    )
    parser.add_argument(
        '--start-frame',
        type=int,
        default=0,
        help='Skip the first N frames (0-indexed).',
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Process at most N frames after --start-frame.',
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=25,
        help='Number of frames per SAM 2 chunk (~12 MB/frame on CPU).',
    )
    parser.add_argument(
        '--no-sam2',
        action='store_true',
        help='Skip SAM 2 entirely and run the red binary model per-frame.',
    )
    parser.add_argument(
        '--blend-epoch61',
        action='store_true',
        help=(
            'Run the red binary model every frame. Where SAM 2 predicts background '
            'but the model is confident about remaining tumor, override SAM 2.'
        ),
    )
    parser.add_argument(
        '--blend-confidence',
        type=float,
        default=0.8,
        help='Minimum red model softmax confidence to trigger override.',
    )
    parser.add_argument(
        '--blend-confidence-low',
        type=float,
        default=None,
        help=(
            'Maximum red model confidence below which a pixel is REMOVED from the '
            'mask, even if SAM 2 (or a prior blend) marked it foreground. Without '
            'this, blending is one-directional -- it can only rescue pixels SAM 2 '
            'missed, never correct pixels SAM 2 wrongly added, so any drift or '
            'smoke-induced false positive is permanent once it appears. Runs every '
            'frame, same as the addition side. Default: disabled (None) for '
            'backward compatibility; try 0.15 to enable.'
        ),
    )
    parser.add_argument(
        '--protect-confidence',
        type=float,
        default=0.3,
        help=(
            'Confidence above which a pixel is added to a PERMANENT, whole-episode '
            'protection map (built only from the baseline model\'s own raw per-frame '
            'predictions, never from SAM 2\'s output). Pixels in this map are exempt '
            'from --blend-confidence-low / --carry-anchor-low removal FOREVER, no '
            'matter how far the toolhead has since moved -- this is what lets '
            'legitimately swept/already-resected regions stay protected without a '
            'fixed distance/radius, and scales automatically with tool speed. '
            'Required for --blend-confidence-low or --carry-anchor-low to be safe '
            'to use with SAM 2 accumulation; try 0.5 (matching --threshold).'
        ),
    )
    parser.add_argument(
        '--sam2-persistence-frames',
        type=int,
        default=None,
        help=(
            'Consecutive frames a pixel must stay in the mask, uncontradicted by the '
            'model (per --blend-confidence-low), before it earns PERMANENT protection '
            '-- same exemption as --protect-confidence, but earned from SAM 2\'s own '
            'sustained tracking rather than the model\'s single-frame confidence. Fixes '
            'the case where SAM 2 is genuinely more accurate than the model (e.g. '
            'early in an episode) and the model itself never validates the region. '
            'Only active during --sam2-persistence-window, to avoid ever locking in '
            'slow drift that happens to survive that many frames later in the episode. '
            'Default: disabled (None); try 25 (~1 chunk at default --chunk-size).'
        ),
    )
    parser.add_argument(
        '--sam2-persistence-window',
        type=int,
        default=50,
        help=(
            'Number of frames (from episode start) during which --sam2-persistence-frames '
            'protection can be newly earned. After this window, no new pixels are added '
            'to the persistence protection map (already-protected pixels stay protected). '
            'Keep well above --sam2-persistence-frames to leave slack for a reset or two. '
            'Only relevant when --sam2-persistence-frames is set. Default: 50.'
        ),
    )
    parser.add_argument(
        '--protect-revoke-frames',
        type=int,
        default=None,
        help=(
            'Consecutive frames of confident model disagreement (per '
            '--blend-confidence-low) required to REVOKE a pixel\'s permanent '
            'protection (--protect-confidence / --sam2-persistence-frames). Without '
            'this, protection is permanent and irrevocable for the rest of the '
            'episode, which blocks legitimate mask retraction when the tool\'s '
            'trajectory genuinely curves away from a region (e.g. an arch-shaped '
            'sweep) -- the model correctly stops predicting it there, but a '
            'validated pixel could never be pruned or dropped from --carry-restore-'
            'protected\'s seed even so. A single low-confidence frame (e.g. a smoke '
            'blip) does NOT revoke anything -- only sustained, consecutive '
            'disagreement does. Default: disabled (None); try 25 (~1 chunk).'
        ),
    )
    parser.add_argument(
        '--revoke-agreement',
        action='store_true',
        help=(
            'Change --protect-revoke-frames from model-confidence-only to AGREEMENT: '
            'a protected pixel is revoked only when the model AND SAM 2 both '
            'confidently call it background, in a CLEAR (non-smoky) scene, for '
            '--protect-revoke-frames consecutive frames. Fixes model-only revoke, '
            'which collapsed to the baseline because the memoryless model reads low '
            'everywhere the tool has left (swept or not). SAM 2\'s memory is the tie-'
            'breaker: it keeps genuinely-swept regions but releases arc-abandoned ones. '
            'Requires --blend-epoch61 and --blend-confidence-low. Also stops '
            '--carry-restore-protected from re-injecting pixels mid-revocation (else '
            'the re-seed would fight the retraction).'
        ),
    )
    parser.add_argument(
        '--revoke-sam2-confidence',
        type=float,
        default=0.5,
        help=(
            'SAM 2 soft-confidence (sigmoid) below which SAM 2 is treated as voting '
            'BACKGROUND for --revoke-agreement. Default 0.5 (SAM 2\'s own decision '
            'boundary); lower to require SAM 2 to be more confidently background.'
        ),
    )
    parser.add_argument(
        '--smoke-saturation',
        type=float,
        default=0.2,
        help=(
            'HSV saturation (0-1) below which a pixel looks smoke-like (desaturated '
            'haze). Combined with --smoke-contrast to build a per-pixel, model-'
            'independent smoke mask that gates --revoke-agreement. Default 0.2.'
        ),
    )
    parser.add_argument(
        '--smoke-contrast',
        type=float,
        default=10.0,
        help=(
            'Local grayscale std-dev (0-255 scale) below which a pixel looks smoke-'
            'like (texture washed out). Combined with --smoke-saturation. Default 10.0.'
        ),
    )
    parser.add_argument(
        '--smoke-window',
        type=int,
        default=7,
        help='Window size (px) for the local-contrast smoke computation. Default 7.',
    )
    parser.add_argument(
        '--warmup-snapshot',
        action='store_true',
        help=(
            'Capture a one-time snapshot of the initial resected region from SAM 2\'s '
            'own confident tracking (NOT gated on model confidence), lock it once the '
            'region is established, and add it to the permanent protection map + '
            '--carry-restore-protected seed. Fixes loss of the initial mask after '
            'aggressive smoke, including regions where the memoryless toolhead model '
            'is confidently wrong but SAM 2 is right (which neither --protect-confidence '
            'nor --sam2-persistence-frames can reach). Requires --blend-epoch61. The '
            'lock trigger is adaptive (growth-rate inflection of SAM 2\'s confident '
            'footprint), not a fixed frame count, so it self-adjusts to session pace.'
        ),
    )
    parser.add_argument(
        '--warmup-sam2-confidence',
        type=float,
        default=0.7,
        help=(
            'SAM 2 soft-confidence (sigmoid of mask logits) above which a pixel counts '
            'toward the warm-up snapshot. Higher = only very confidently-tracked pixels '
            'are captured. Default 0.7.'
        ),
    )
    parser.add_argument(
        '--warmup-clarity-confidence',
        type=float,
        default=0.6,
        help=(
            'Mean per-frame model confidence below which a frame is treated as unclear '
            '(smoke/occlusion) and SKIPPED for warm-up accumulation, so early smoke '
            'cannot corrupt the snapshot. Default 0.6.'
        ),
    )
    parser.add_argument(
        '--warmup-persist-frames',
        type=int,
        default=3,
        help=(
            'Consecutive clear frames SAM 2 must hold a pixel (at >= '
            '--warmup-sam2-confidence) before it enters the snapshot candidate, to '
            'reject one-frame flickers. Default 3.'
        ),
    )
    parser.add_argument(
        '--warmup-inflection-ratio',
        type=float,
        default=0.2,
        help=(
            'Lock the snapshot once the candidate\'s recent growth (new px/clear-frame, '
            'smoothed) falls below this fraction of its peak growth -- i.e. SAM 2 has '
            'finished establishing the initial region. Session-invariant (a ratio, not '
            'an absolute count). Default 0.2.'
        ),
    )
    parser.add_argument(
        '--warmup-min-frames',
        type=int,
        default=10,
        help=(
            'Minimum clear frames observed before the inflection lock is allowed to '
            'fire, so the snapshot cannot lock on the first frame or two. Default 10.'
        ),
    )
    parser.add_argument(
        '--warmup-max-frames',
        type=int,
        default=100,
        help=(
            'Fallback cap: force the snapshot to lock at this absolute frame index if '
            'the growth-rate inflection never fires (e.g. a session that sweeps '
            'continuously with no distinct establishment phase). Default 100.'
        ),
    )
    parser.add_argument(
        '--reassert-protected',
        action='store_true',
        help=(
            'Every frame, force the already-accumulated protected pixels '
            '(--protect-confidence, minus any revoked/suspended) back into the mask, '
            'even if SAM 2 dropped them mid-chunk. Fixes sudden mask disappearance '
            'caused by SAM 2 shrinking its tracked region between chunk re-seeds '
            '(carry-restore only re-injects at chunk boundaries). Makes the '
            'accumulated region persistent frame-to-frame; revoke still lets the arc '
            'retract since revoked pixels leave the protected set.'
        ),
    )
    parser.add_argument(
        '--removal-hysteresis',
        type=int,
        default=1,
        help=(
            'Consecutive frames a pixel must stay below --blend-confidence-low before '
            'the symmetric removal strips it. Default 1 = strip on any single low '
            'frame (legacy, but a momentary confidence dip from tool motion/blur then '
            'collapses the mask, since unprotected pixels are removed instantly). '
            'Higher (e.g. 8-15) ignores transient dips and only removes genuinely '
            'sustained false positives -- directly prevents the sudden mask collapse.'
        ),
    )
    parser.add_argument(
        '--revoke-mode',
        choices=['delete', 'suspend'],
        default='delete',
        help=(
            'What revocation does to a protected pixel. "delete" (default) removes it '
            'from the protection map permanently -- irreversible, so a single smoke '
            'misdetection destroys accumulated history forever. "suspend" instead '
            'temporarily un-protects it but lets it REJOIN protection the moment the '
            'model re-confirms it (conf >= --protect-confidence) -- so a false '
            'revoke during undetected smoke self-heals when the scene clears. '
            'Addresses the root cause of catastrophic smoke loss, not just the trigger.'
        ),
    )
    parser.add_argument(
        '--limit-lateral-growth',
        action='store_true',
        help=(
            'Restrict the mask from growing sideways into tumor regions the toolhead '
            'has not swept. Maintains an accumulated "swept" baseline; each frame the '
            'mask may only occupy that baseline plus a vertical slab around the '
            'toolhead\'s current horizontal position (estimated from the top of the '
            'model prediction). New area outside that (lateral drift) is clipped. '
            'Never touches already-swept mask, so it does not cut good regions. '
            'Targets model-supported lateral over-expansion that confidence thresholds '
            'cannot (the model confidently predicts the un-swept region). Requires '
            '--blend-epoch61.'
        ),
    )
    parser.add_argument(
        '--growth-frontier-margin',
        type=int,
        default=40,
        help=(
            'Horizontal margin (px) beyond the toolhead\'s current x-extent within '
            'which new lateral growth is allowed (--limit-lateral-growth). Smaller = '
            'tighter clip. Default 40.'
        ),
    )
    parser.add_argument(
        '--growth-frontier-band',
        type=int,
        default=60,
        help=(
            'Height (px) of the top band of the model prediction used to estimate the '
            'toolhead\'s current horizontal position (--limit-lateral-growth). Default 60.'
        ),
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help='Confidence threshold for red class (default 0.6). Lower to 0.4-0.5 for difficult/low-contrast videos.',
    )
    parser.add_argument(
        '--gaussian-sigma',
        type=float,
        default=0.0,
        help=(
            'Apply Gaussian blur to the probability map before thresholding (0 = disabled). '
            'Smooths spatial noise at boundaries, reducing flickering. Try 2.0–5.0.'
        ),
    )
    parser.add_argument(
        '--blend-ema-alpha',
        type=float,
        default=1.0,
        help=(
            'EMA smoothing factor for red model probabilities across frames. '
            '1.0 = no smoothing. 0.2 = heavy smoothing.'
        ),
    )
    parser.add_argument(
        '--blend-carry-frames',
        type=int,
        default=1,
        help=(
            'Number of recent blended frames to majority-vote for the carry mask. '
            '1 = last frame only. 3-5 = more robust to noise.'
        ),
    )
    parser.add_argument(
        '--blend-carry-weighted',
        action='store_true',
        help='Weight carry buffer frames by model confidence instead of uniform voting.',
    )
    parser.add_argument(
        '--carry-anchor',
        action='store_true',
        help=(
            'At each chunk boundary, anchor the carry against the red model\'s current '
            'prediction. Restores pixels the model is very confident about that SAM 2 '
            'dropped after movement. Requires --blend-epoch61.'
        ),
    )
    parser.add_argument(
        '--carry-anchor-high',
        type=float,
        default=0.7,
        help=(
            'Red model confidence above which a pixel is added to carry regardless of '
            'majority vote. Restores regions SAM 2 dropped after movement.'
        ),
    )
    parser.add_argument(
        '--carry-anchor-low',
        type=float,
        default=None,
        help=(
            'Red model confidence below which a pixel is REMOVED from the carry '
            '(the seed handed to the next chunk\'s SAM 2 init), even if it was in '
            'the majority-vote carry. Mirrors --carry-anchor-high on the removal '
            'side -- without it, carry-anchor is also one-directional and can only '
            'rescue pixels, never prune accumulated drift from the seed. Default: '
            'disabled (None); try 0.2 to enable.'
        ),
    )
    parser.add_argument(
        '--carry-restore-protected',
        action='store_true',
        help=(
            'At each chunk boundary, re-inject validated_union / persistent_union '
            '(--protect-confidence / --sam2-persistence-frames) into the seed handed '
            'to the next chunk\'s SAM 2 init. Fixes the opposite failure from removal: '
            'SAM 2 outright losing an already-validated region (e.g. after a smoke-out '
            'or occlusion causes a whole chunk of tracking to fail), which the removal-'
            'side exemptions do nothing for since they only ever protect pixels SAM 2 '
            'currently has as foreground -- they never restore ones it dropped. Safe '
            'against the deformation/drift problem that sank monotonic accumulation '
            'earlier: this only injects at chunk boundaries (not every frame), and SAM '
            '2\'s own propagation re-tracks the injected region against the actual video '
            'for the rest of the chunk, rather than pasting frozen coordinates forever.'
        ),
    )
    parser.add_argument(
        '--green-point',
        type=str,
        default=None,
        help='Track charred cut line as SAM 2 object (green). Provide "x,y" coordinates.',
    )
    parser.add_argument(
        '--green-point-frame',
        type=int,
        default=0,
        help='Frame index (0-based) where --green-point is located.',
    )
    parser.add_argument(
        '--green-model',
        type=str,
        default=None,
        help='Path to binary green model checkpoint (.pth) for per-frame green detection.',
    )
    parser.add_argument(
        '--green-metadata',
        type=str,
        default=None,
        help='Path to binary green model metadata (.json). Required with --green-model.',
    )
    parser.add_argument(
        '--green-confidence',
        type=float,
        default=0.5,
        help='Softmax probability threshold for the binary green model.',
    )
    parser.add_argument(
        '--debug-blend',
        action='store_true',
        help='Print red model vs SAM 2 diagnostics for first 10 frames.',
    )
    parser.add_argument(
        '--fill-gap',
        action='store_true',
        help='Dilate red into adjacent background pixels to fill boundary gaps.',
    )
    parser.add_argument(
        '--fill-gap-kernel',
        type=int,
        default=5,
        help='Ellipse kernel size (pixels) for gap-fill dilation.',
    )
    parser.add_argument(
        '--color',
        type=str,
        default='255,0,0',
        help='RGB colour for the primary mask (default red). Use "0,0,255" for blue.',
    )
    parser.add_argument(
        '--reinit-every',
        type=int,
        default=0,
        help=(
            'Re-seed SAM 2 from a fresh model prediction every N frames (at chunk '
            'boundaries). 0 = disabled. Use with --chunk-size to control granularity, '
            'e.g. --reinit-every 30 --chunk-size 30 re-seeds every 30 frames.'
        ),
    )
    parser.add_argument(
        '--spatial-model',
        type=str,
        default='checkpoints_tumor_binary10/best_model_epoch_swa.pth',
        help=(
            'Path to a multi-class model (e.g. epoch 60) used as a spatial constraint. '
            'Any pixel predicted as non-background by this model defines the allowed '
            'tumor region. Red predictions outside this region are suppressed. '
            'Defaults to binary10 SWA; pass --no-spatial-constraint to disable.'
        ),
    )
    parser.add_argument(
        '--no-spatial-constraint',
        action='store_true',
        help='Disable the spatial constraint model entirely (overrides --spatial-model).',
    )
    parser.add_argument(
        '--spatial-metadata',
        type=str,
        default='checkpoints_tumor_binary10/best_model_epoch_swa_metadata.json',
        help='Metadata JSON for --spatial-model.',
    )
    parser.add_argument(
        '--spatial-dilate',
        type=int,
        default=40,
        help=(
            'Dilation (pixels) applied to the spatial constraint mask before ANDing. '
            'Ensures the red margin at the tumor boundary is not clipped.'
        ),
    )
    parser.add_argument(
        '--tta',
        action='store_true',
        help=(
            '5-rotation TTA (0, ±10°, ±15°) — in-distribution augmentations for models '
            'trained with ±20° rotation. Averages 5 predictions per frame. ~5x slower. '
            'Only applies to --no-sam2 mode.'
        ),
    )
    parser.add_argument(
        '--spatial-mask-dir',
        type=str,
        default=None,
        help=(
            'Directory of pre-computed spatial constraint masks (e.g. output of binary3+SAM2 run). '
            'Each frame_mask.png is matched by stem to the corresponding input frame, binarised '
            '(any non-black pixel = tumor), dilated by --spatial-dilate, then ANDed with the '
            'margin model output. Use instead of --spatial-model when the spatial constraint '
            'was generated in a separate pass (e.g. SAM2-propagated tumor masks).'
        ),
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default=os.environ.get('SAM2_SCRATCH', os.path.join(tempfile.gettempdir(), 'sam2_scratch')),
        help=(
            'Directory for SAM 2\'s per-chunk JPEG scratch files. Defaults to E: '
            '(NOT the system temp dir on C:) -- SAM 2\'s video predictor requires '
            'chunk frames written to disk, and Python\'s tempfile module defaults '
            'to C:\\Users\\...\\AppData\\Local\\Temp if unspecified, which can fill '
            'a nearly-full C: drive during long batches. Auto-created if missing.'
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()

    try:
        mask_color = tuple(int(x) for x in args.color.split(','))
        assert len(mask_color) == 3
    except Exception:
        print(f"Error: --color must be R,G,B (e.g. 0,0,255), got: {args.color}")
        sys.exit(1)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'

    sam2_device = args.sam2_device
    if sam2_device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available for SAM 2, falling back to CPU")
        sam2_device = 'cpu'

    if args.no_spatial_constraint:
        args.spatial_model = None
        args.spatial_metadata = None

    print("=" * 70)
    print("SAM 2 + Red Binary Model Hybrid Tumor Segmentation Inference")
    print("=" * 70)
    print(f"Model:        {args.model}")
    if args.spatial_model:
        print(f"Spatial constraint: {args.spatial_model} (dilate={args.spatial_dilate}px)")
    if args.green_model:
        print(f"Green model:  {args.green_model} (threshold={args.green_confidence})")
    print(f"Input:        {args.input}")
    print(f"Output:       {args.output_dir}")
    print(f"UNet device:  {device}")
    print(f"SAM2 device:  {sam2_device}")
    print(f"Visualize:    {args.visualize}")
    print()

    try:
        # ---- Collect frames --------------------------------------------------
        input_dir = Path(args.input)
        if not input_dir.exists():
            print(f"Error: input directory not found: {input_dir}")
            sys.exit(1)

        frame_paths = collect_frames(input_dir)
        if not frame_paths:
            print(f"Error: no image files found in {input_dir}")
            sys.exit(1)

        if args.start_frame:
            frame_paths = frame_paths[args.start_frame:]
        if args.max_frames is not None:
            frame_paths = frame_paths[:args.max_frames]
        print(f"Found {len(frame_paths)} frames" +
              (f" (starting at frame {args.start_frame + 1})" if args.start_frame else "") +
              (f", capped at {args.max_frames}" if args.max_frames else "") + ".")

        # ---- No-SAM2 mode: pure per-frame inference -------------------------
        if args.no_sam2:
            print("\nSAM 2 disabled — running red binary model per-frame.")
            if args.blend_ema_alpha < 1.0:
                print(f"EMA smoothing enabled (alpha={args.blend_ema_alpha}).")
            if args.tta:
                print("TTA enabled (5-rotation: 0, ±10°, ±15°) — ~5x slower per frame.")
            engine = InferenceEngine(
                model_path=args.model,
                metadata_path=args.metadata,
                device=device,
            )
            spatial_engine = None
            spatial_mask_lookup = None
            if args.spatial_mask_dir:
                print(f"Loading pre-computed spatial masks from: {args.spatial_mask_dir}")
                spatial_mask_lookup = load_precomputed_spatial_masks(
                    Path(args.spatial_mask_dir), frame_paths
                )
            elif args.spatial_model:
                if not args.spatial_metadata:
                    print("Error: --spatial-metadata is required with --spatial-model.")
                    sys.exit(1)
                print("Loading spatial constraint model...")
                spatial_engine = InferenceEngine(
                    model_path=args.spatial_model,
                    metadata_path=args.spatial_metadata,
                    device=device,
                )
                print("Spatial constraint model loaded.")
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            viz_generator = VisualizationGenerator() if args.visualize else None
            first_img = np.array(Image.open(frame_paths[0]).convert('RGB'))
            H_orig, W_orig = first_img.shape[:2]

            import torch.nn.functional as F
            ema_probs = None
            t0 = time.perf_counter()

            for i, fp in enumerate(frame_paths, 1):
                img = np.array(Image.open(fp).convert('RGB'))
                red_prob_512 = predict_red_512(engine, img, tta=args.tta)  # (512, 512) float32

                conf = F.interpolate(
                    torch.from_numpy(red_prob_512).unsqueeze(0).unsqueeze(0),
                    size=(H_orig, W_orig),
                    mode='bilinear', align_corners=False,
                ).squeeze().cpu().numpy()

                if args.blend_ema_alpha < 1.0:
                    if ema_probs is None:
                        ema_probs = conf.copy()
                    else:
                        ema_probs = (args.blend_ema_alpha * conf
                                     + (1.0 - args.blend_ema_alpha) * ema_probs)
                    conf_for_thresh = ema_probs
                else:
                    conf_for_thresh = conf

                if args.gaussian_sigma > 0:
                    from scipy.ndimage import gaussian_filter
                    conf_for_thresh = gaussian_filter(conf_for_thresh, sigma=args.gaussian_sigma)

                remaining_mask = conf_for_thresh >= args.threshold

                if spatial_mask_lookup is not None:
                    spatial_mask = get_precomputed_spatial_mask(
                        spatial_mask_lookup, fp, H_orig, W_orig, args.spatial_dilate
                    )
                    remaining_mask = remaining_mask & spatial_mask
                elif spatial_engine is not None:
                    spatial_mask = get_spatial_mask(
                        spatial_engine, img, H_orig, W_orig, args.spatial_dilate
                    )
                    remaining_mask = remaining_mask & spatial_mask

                rgb_mask = compose_rgb_mask(remaining_mask, H_orig, W_orig, color=mask_color)

                Image.fromarray(rgb_mask.astype(np.uint8)).save(
                    output_dir / f'{fp.stem}_mask.png')
                Image.open(fp).save(output_dir / f'{fp.stem}_org.png')
                if viz_generator is not None:
                    overlay = viz_generator.generate_overlay(img, rgb_mask, alpha=args.alpha)
                    Image.fromarray(overlay.astype(np.uint8)).save(
                        output_dir / f'{fp.stem}_overlay.png')
                print(f"  [{i}/{len(frame_paths)}] {fp.name}")

            elapsed = time.perf_counter() - t0
            fps = len(frame_paths) / elapsed
            print(f"\n{'=' * 70}")
            print(f"Frames processed:  {len(frame_paths)}")
            print(f"Time:              {elapsed:.1f}s")
            print(f"FPS:               {fps:.1f}")
            print(f"Output:            {output_dir}")
            return

        # ---- Load SAM 2 first -----------------------------------------------
        print("\nLoading SAM 2...")
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            print(
                "SAM 2 is not installed.\n"
                "Install it with:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git"
            )
            sys.exit(1)

        sam2_checkpoint = ensure_sam2_checkpoint(Path(args.sam2_checkpoint_dir))

        from hydra import initialize_config_module
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        with initialize_config_module('sam2', version_base='1.2'):
            sam2_predictor = build_sam2_video_predictor(
                SAM2_CONFIG, str(sam2_checkpoint), device=sam2_device
            )
        print(f"SAM 2 loaded on {sam2_device}.")

        # ---- Load red binary model ------------------------------------------
        print("\nLoading red binary model...")
        engine = InferenceEngine(
            model_path=args.model,
            metadata_path=args.metadata,
            device=device,
        )
        print("Red binary model loaded.")

        # Load binary green model if specified
        green_engine = None
        if args.green_model:
            if not args.green_metadata:
                print("Error: --green-metadata is required when --green-model is set.")
                sys.exit(1)
            print("\nLoading binary green model...")
            green_engine = InferenceEngine(
                model_path=args.green_model,
                metadata_path=args.green_metadata,
                device=device,
            )
            print(f"Binary green model loaded (threshold={args.green_confidence}).")

        # Load spatial constraint model if specified
        spatial_engine = None
        if args.spatial_model:
            if not args.spatial_metadata:
                print("Error: --spatial-metadata is required with --spatial-model.")
                sys.exit(1)
            print("\nLoading spatial constraint model...")
            spatial_engine = InferenceEngine(
                model_path=args.spatial_model,
                metadata_path=args.spatial_metadata,
                device=device,
            )
            print("Spatial constraint model loaded.")

        # ---- Run on frame 1 for SAM 2 initialisation ------------------------
        print("\nRunning red binary model on frame 1 for SAM 2 initialisation...")
        first_img = np.array(Image.open(frame_paths[0]).convert('RGB'))
        H_orig, W_orig = first_img.shape[:2]

        probs, _ = engine.predict_single(first_img)          # (2, 512, 512)
        class_map_512 = probs.argmax(dim=0).cpu().numpy()    # (512, 512)

        class_map_orig = np.array(
            Image.fromarray(class_map_512.astype(np.uint8)).resize(
                (W_orig, H_orig), Image.NEAREST
            )
        )

        remaining_init = (class_map_orig == 1)
        has_remaining  = bool(remaining_init.any())

        print(
            f"  Remaining tumor (red):  {remaining_init.sum():>7,} px  — "
            f"{'found' if has_remaining else 'NONE'}"
        )

        import gc
        needs_engine = args.blend_epoch61 or green_engine is not None or args.reinit_every > 0

        # ---- Output setup ---------------------------------------------------
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_generator = VisualizationGenerator() if args.visualize else None

        t_total_start = time.perf_counter()
        frames_written = 0
        n_empty        = 0
        n_blend_removed_px    = 0   # symmetric-blend removal: total px stripped, per-frame
        n_blend_removed_frames = 0  # frames where the per-frame removal fired at all
        n_carry_removed_px    = 0   # carry-anchor-low removal: total px stripped, at chunk boundaries
        n_carry_removed_frames = 0  # chunk boundaries where the carry removal fired at all
        n_persistence_newly_protected = 0  # px added to persistent_union this episode
        n_carry_restored_px    = 0   # carry-restore-protected: total px re-injected at chunk boundaries
        n_carry_restored_frames = 0  # chunk boundaries where the restore fired at all
        n_protection_revoked_px = 0  # protect-revoke-frames: total px un-protected this episode
        n_lateral_clipped_px = 0     # limit-lateral-growth: total px clipped as lateral drift
        n_lateral_clipped_frames = 0 # frames where the lateral clip fired
        warmup_lock_frame = None     # absolute frame index where warm-up snapshot locked
        warmup_lock_reason = None    # 'inflection' or 'max-frames' or None

        # ---- Scan for first red frame ----------------------------------------
        # Write per-frame outputs until the model first detects red, then hand
        # off to SAM 2 from that frame onward. This handles videos that begin
        # before any cuts have been made.
        if not has_remaining:
            print("\nNo red on frame 1 — scanning forward for first frame with tumor...")
            for scan_idx, fp in enumerate(frame_paths):
                img_s = np.array(Image.open(fp).convert('RGB'))
                probs_s, _ = engine.predict_single(img_s)
                cm_s = np.array(
                    Image.fromarray(
                        probs_s.argmax(dim=0).cpu().numpy().astype(np.uint8)
                    ).resize((W_orig, H_orig), Image.NEAREST)
                )
                red_s = (cm_s == 1)

                if red_s.any():
                    remaining_init = red_s
                    has_remaining  = True
                    frame_paths    = frame_paths[scan_idx:]
                    print(f"  → Red first detected at frame {scan_idx + 1} ({fp.name}). "
                          f"SAM 2 initialising from here.")
                    break

                # No red yet — write per-frame output and keep scanning
                rgb_s = compose_rgb_mask(red_s, H_orig, W_orig, color=mask_color)
                Image.fromarray(rgb_s.astype(np.uint8)).save(
                    output_dir / f'{fp.stem}_mask.png')
                Image.open(fp).save(output_dir / f'{fp.stem}_org.png')
                if viz_generator is not None:
                    overlay = viz_generator.generate_overlay(img_s, rgb_s, alpha=args.alpha)
                    Image.fromarray(overlay.astype(np.uint8)).save(
                        output_dir / f'{fp.stem}_overlay.png')
                frames_written += 1

            if not has_remaining:
                print("\nNo red detected in any frame. Per-frame inference complete.")
                total_elapsed = time.perf_counter() - t_total_start
                print(f"\n{'=' * 70}")
                print(f"Frames processed:  {frames_written}")
                print(f"Time:              {total_elapsed:.1f}s")
                print(f"Output:            {output_dir}")
                return

        # Free engine if not needed for blend/green
        if not needs_engine:
            del engine.model
            del engine
            gc.collect()
            if device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        else:
            reasons = []
            if args.blend_epoch61:
                reasons.append(f"blend (confidence={args.blend_confidence})")
            if green_engine is not None:
                reasons.append("green model")
            if args.reinit_every > 0:
                reasons.append(f"reinit every {args.reinit_every} frames")
            print(f"Keeping model alive for: {', '.join(reasons)}.")

        # ---- Chunked SAM 2 processing ---------------------------------------
        chunk_size = args.chunk_size
        n_chunks = (len(frame_paths) + chunk_size - 1) // chunk_size
        print(
            f"\nProcessing {len(frame_paths)} frames in {n_chunks} chunk(s) "
            f"of {chunk_size} frames each "
            f"(~{chunk_size * 12:.0f} MB CPU RAM per chunk)."
        )

        carry_remaining = remaining_init
        carry_green     = np.zeros((H_orig, W_orig), dtype=bool)
        green_prompted  = False

        # Persistent protection map: union of every pixel the baseline model has
        # EVER confidently validated, across the whole episode (not per-chunk).
        # Built only from the model's own raw per-frame predictions -- never from
        # SAM 2's output -- so it can't be corrupted by SAM 2's own drift. Used to
        # exempt legitimately-swept, already-resected regions from symmetric
        # removal, regardless of how far the toolhead has since moved. Scales with
        # tool speed/history automatically (no fixed distance/radius parameter).
        validated_union = np.zeros((H_orig, W_orig), dtype=bool)

        # Second, independent protection map: pixels SAM 2 has held stably for
        # --sam2-persistence-frames consecutive frames without the model actively
        # contradicting them, earned only during --sam2-persistence-window at the
        # start of the episode. Covers the case validated_union can't: SAM 2's own
        # tracking being genuinely more accurate than the model's single-frame call
        # (observed early in episodes), where the model never independently
        # validates the region at --protect-confidence. Bounded to the settling
        # window on purpose -- unbounded, this same mechanism could eventually lock
        # in slow static-scene drift if it happened to go unresolved for that long.
        persistence_counter = np.zeros((H_orig, W_orig), dtype=np.int32)
        persistent_union    = np.zeros((H_orig, W_orig), dtype=bool)

        # Counter for revoking permanent protection after sustained contradiction
        # (--protect-revoke-frames). Not window-limited like persistence -- a
        # trajectory can curve away from a region at any point in the episode.
        revoke_counter = np.zeros((H_orig, W_orig), dtype=np.int32)
        # Suspended pixels (--revoke-mode suspend): temporarily un-protected by
        # revocation, but eligible to rejoin protection when the model re-confirms.
        suspended = np.zeros((H_orig, W_orig), dtype=bool)
        # Per-pixel counter of consecutive frames below --blend-confidence-low, for
        # --removal-hysteresis (don't strip on a transient dip).
        low_conf_counter = np.zeros((H_orig, W_orig), dtype=np.int32)

        # Accumulated "swept" baseline for --limit-lateral-growth: monotonic union of
        # every mask region legitimately accepted so far. New growth is only allowed
        # within this baseline plus a slab around the toolhead's current x-position,
        # so the mask can't creep laterally into un-swept tumor.
        growth_baseline = remaining_init.copy()

        # Warm-up snapshot state (--warmup-snapshot): a one-time capture of the
        # initial resected region from SAM 2's own confident tracking, locked once
        # the region stops growing (growth-rate inflection). Built during CLEAR
        # frames only so early smoke can't corrupt it. Once locked, it joins the
        # permanent protection map and the carry-restore seed.
        warmup_snapshot   = np.zeros((H_orig, W_orig), dtype=bool)  # locked result
        warmup_candidate  = np.zeros((H_orig, W_orig), dtype=bool)  # accumulating
        warmup_hold_ctr   = np.zeros((H_orig, W_orig), dtype=np.int32)  # anti-flicker
        warmup_locked     = False
        warmup_clear_seen = 0        # count of clear frames observed pre-lock
        warmup_peak_growth = 0.0     # peak smoothed growth (new px / clear frame)
        warmup_growth_win: list = [] # rolling window of recent per-clear-frame growth

        green_xy = None
        if args.green_point:
            try:
                gx, gy = map(int, args.green_point.split(','))
                green_xy = (gx, gy)
                print(f"Green point prompt: ({gx}, {gy}) on frame {args.green_point_frame}")
            except ValueError:
                print(f"Warning: could not parse --green-point '{args.green_point}'. Expected 'x,y'.")

        # EMA state for red model probability smoothing (persists across chunks)
        ema_probs: Optional[np.ndarray] = None

        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end   = min(chunk_start + chunk_size, len(frame_paths))
            chunk_paths = frame_paths[chunk_start:chunk_end]

            print(
                f"  Chunk {chunk_idx + 1}/{n_chunks}: "
                f"frames {chunk_start + 1}–{chunk_end} ..."
            )

            scratch_root = Path(args.temp_dir)
            scratch_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=str(scratch_root)) as tmp_dir:
                tmp_path = Path(tmp_dir)

                for j, fp in enumerate(chunk_paths):
                    Image.open(fp).convert('RGB').save(
                        tmp_path / f"{j:06d}.jpg", 'JPEG', quality=95
                    )

                inference_state = sam2_predictor.init_state(
                    video_path=str(tmp_path),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                )

                obj_ids_added = []
                if carry_remaining.any():
                    sam2_predictor.add_new_mask(
                        inference_state, frame_idx=0, obj_id=1,
                        mask=carry_remaining,
                    )
                    obj_ids_added.append(1)

                # Green: mask carry for subsequent chunks, point prompt for first occurrence
                if green_xy is not None:
                    if carry_green.any():
                        sam2_predictor.add_new_mask(
                            inference_state, frame_idx=0, obj_id=3,
                            mask=carry_green,
                        )
                        obj_ids_added.append(3)
                    elif not green_prompted and chunk_start <= args.green_point_frame < chunk_end:
                        rel_idx = args.green_point_frame - chunk_start
                        sam2_predictor.add_new_points_or_box(
                            inference_state,
                            frame_idx=rel_idx,
                            obj_id=3,
                            points=np.array([[green_xy[0], green_xy[1]]], dtype=np.float32),
                            labels=np.array([1], dtype=np.int32),
                        )
                        obj_ids_added.append(3)
                        green_prompted = True
                        print(f"    [green] Point prompt injected at frame {args.green_point_frame} "
                              f"(chunk-relative idx {rel_idx})")

                if not obj_ids_added:
                    sam2_predictor.reset_state(inference_state)
                    chunk_segs = {}
                    chunk_sam2_conf = {}
                else:
                    chunk_segs = {}
                    chunk_sam2_conf = {}
                    for out_idx, out_obj_ids, out_logits in \
                            sam2_predictor.propagate_in_video(inference_state):
                        chunk_segs[out_idx] = {
                            oid: (out_logits[k, 0] > 0.0).cpu().numpy()
                            for k, oid in enumerate(out_obj_ids)
                        }
                        # Keep SAM 2's soft confidence (sigmoid of logits), not just the
                        # binarized mask -- --warmup-snapshot needs it to capture regions
                        # SAM 2 is genuinely confident about, independent of the model.
                        chunk_sam2_conf[out_idx] = {
                            oid: torch.sigmoid(out_logits[k, 0]).cpu().numpy()
                            for k, oid in enumerate(out_obj_ids)
                        }
                    sam2_predictor.reset_state(inference_state)
                    if sam2_device == 'cuda':
                        torch.cuda.empty_cache()

                # Update carry-over from last frame of this chunk
                if chunk_segs:
                    last_seg = chunk_segs.get(len(chunk_paths) - 1, {})
                    carry_remaining = last_seg.get(1, np.zeros((H_orig, W_orig), dtype=bool))
                    carry_green     = last_seg.get(3, np.zeros((H_orig, W_orig), dtype=bool))
                else:
                    carry_remaining = np.zeros((H_orig, W_orig), dtype=bool)
                    carry_green     = np.zeros((H_orig, W_orig), dtype=bool)

                # Write outputs for this chunk
                blended_last_remaining = None
                blended_buffer_rem: list = []
                blended_buffer_weights: list = []
                _frame_conf = 1.0

                for j, fp in enumerate(chunk_paths):
                    seg = chunk_segs.get(j, {})
                    remaining_mask = seg.get(1, np.zeros((H_orig, W_orig), dtype=bool))
                    sam2_conf_map = chunk_sam2_conf.get(j, {}).get(
                        1, np.zeros((H_orig, W_orig), dtype=np.float32)
                    )

                    # Blend: override where red model is confident but SAM 2 missed
                    if args.blend_epoch61:
                        import torch.nn.functional as F
                        img_arr = np.array(Image.open(fp).convert('RGB'))
                        probs, _ = engine.predict_single(img_arr)   # (2, 512, 512)

                        _frame_conf = float(probs.max(dim=0).values.mean().cpu())

                        conf = probs[1].unsqueeze(0).unsqueeze(0)   # (1,1,512,512)
                        conf_orig = F.interpolate(
                            conf, size=(H_orig, W_orig),
                            mode='bilinear', align_corners=False
                        ).squeeze().cpu().numpy()

                        if args.blend_ema_alpha < 1.0:
                            if ema_probs is None:
                                ema_probs = conf_orig.copy()
                            else:
                                ema_probs = (args.blend_ema_alpha * conf_orig
                                             + (1.0 - args.blend_ema_alpha) * ema_probs)
                            conf_for_thresh = ema_probs
                        else:
                            conf_for_thresh = conf_orig

                        # Grow the permanent protection map from this frame's (EMA-smoothed)
                        # confidence. Uses --protect-confidence, not --blend-confidence, so
                        # it can be tuned independently of the add-threshold.
                        if args.protect_confidence is not None:
                            validated_union |= (conf_for_thresh >= args.protect_confidence)
                            # Suspend-mode self-heal: a suspended pixel rejoins protection
                            # the moment the model is confident about it again (e.g. once
                            # smoke clears), undoing a false revoke.
                            if args.revoke_mode == 'suspend':
                                suspended &= ~(conf_for_thresh >= args.protect_confidence)

                        new_red = (~remaining_mask) & (conf_for_thresh >= args.blend_confidence)

                        if args.debug_blend and new_red.any():
                            if not hasattr(args, '_debug_count'):
                                args._debug_count = 0
                            if args._debug_count < 10:
                                args._debug_count += 1
                                n_new = new_red.sum()
                                n_sam2_bg  = (~remaining_mask & new_red).sum()
                                n_sam2_red = (remaining_mask & new_red).sum()
                                print(f"\n[DEBUG blend] Frame {fp.name}  (new_red: {n_new}px)")
                                print(f"  SAM2 in new_red region → "
                                      f"background:{n_sam2_bg}px  red:{n_sam2_red}px")
                                conf_vals = conf_for_thresh[new_red]
                                print(f"  red model conf → "
                                      f"min:{conf_vals.min():.3f}  "
                                      f"mean:{conf_vals.mean():.3f}  "
                                      f"max:{conf_vals.max():.3f}")

                        remaining_mask = remaining_mask | new_red

                        # Persistence tracking: a pixel's counter increments while it
                        # stays in the mask AND the model isn't actively contradicting
                        # it (same threshold as the removal below); any lapse resets it
                        # to 0. Reaching --sam2-persistence-frames earns permanent
                        # protection. Gated to the first --sam2-persistence-window
                        # frames of the episode only -- see note at persistent_union init.
                        if args.sam2_persistence_frames is not None:
                            global_frame_idx = chunk_start + j
                            if global_frame_idx < args.sam2_persistence_window:
                                not_contradicted = (
                                    True if args.blend_confidence_low is None
                                    else (conf_for_thresh >= args.blend_confidence_low)
                                )
                                still_tracking = remaining_mask & not_contradicted
                                persistence_counter = np.where(
                                    still_tracking, persistence_counter + 1, 0
                                )
                                newly_protected = (
                                    (persistence_counter >= args.sam2_persistence_frames)
                                    & ~persistent_union
                                )
                                n_persistence_newly_protected += int(newly_protected.sum())
                                persistent_union |= newly_protected

                        # Revoke permanent protection after sustained contradiction, so a
                        # genuinely-curving (arc) trajectory can retract the mask instead
                        # of the region staying un-prunable forever. Two modes:
                        #   * model-only (default): the model confidently disagrees. NOTE
                        #     this collapses to the baseline for a memoryless model, which
                        #     reads low everywhere the tool has left -- kept only for
                        #     backward compat.
                        #   * --revoke-agreement: the model AND SAM 2 both confidently call
                        #     the pixel background, in a CLEAR scene. SAM 2's memory is the
                        #     tie-breaker -- it keeps genuinely-swept regions but releases
                        #     arc-abandoned ones, so this does NOT collapse to baseline.
                        # A single smoke-blip frame never revokes -- only sustained,
                        # consecutive agreement does.
                        if args.protect_revoke_frames is not None and args.blend_confidence_low is not None:
                            if args.revoke_agreement:
                                import cv2
                                model_bg = conf_for_thresh < args.blend_confidence_low
                                sam2_bg  = sam2_conf_map < args.revoke_sam2_confidence
                                # Per-pixel, model-INDEPENDENT smoke mask: desaturated haze
                                # (low HSV saturation) with washed-out texture (low local
                                # contrast). Gates removal locally so smoke over the region
                                # can't be masked by a globally-clear frame, and so the
                                # clarity gate isn't just the model's own confidence again.
                                hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
                                sat = hsv[:, :, 1].astype(np.float32) / 255.0
                                gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
                                w = args.smoke_window | 1
                                lmean = cv2.boxFilter(gray, -1, (w, w))
                                lsq   = cv2.boxFilter(gray * gray, -1, (w, w))
                                lstd  = np.sqrt(np.maximum(lsq - lmean * lmean, 0.0))
                                is_smoky = (sat < args.smoke_saturation) & (lstd < args.smoke_contrast)
                                contradicted = model_bg & sam2_bg & ~is_smoky
                            else:
                                contradicted = conf_for_thresh < args.blend_confidence_low
                            revoke_counter = np.where(contradicted, revoke_counter + 1, 0)
                            revoke_now = (revoke_counter >= args.protect_revoke_frames) & (
                                (validated_union | persistent_union | warmup_snapshot)
                                & ~suspended
                            )
                            if revoke_now.any():
                                n_protection_revoked_px += int(revoke_now.sum())
                                if args.revoke_mode == 'suspend':
                                    # Reversible: mark suspended, keep in the maps so it
                                    # can rejoin protection when the model re-confirms.
                                    suspended = suspended | revoke_now
                                else:
                                    # Permanent deletion (legacy).
                                    validated_union  = validated_union  & ~revoke_now
                                    persistent_union = persistent_union & ~revoke_now
                                    warmup_snapshot  = warmup_snapshot  & ~revoke_now
                                remaining_mask   = remaining_mask   & ~revoke_now
                                revoke_counter   = np.where(revoke_now, 0, revoke_counter)

                        # Warm-up snapshot: accumulate SAM 2's confidently-tracked pixels
                        # during clear frames, then lock once the confident footprint stops
                        # growing (establishment complete). Captured from SAM 2's own soft
                        # confidence, NOT the model -- so it reaches regions the memoryless
                        # model is confidently wrong about but SAM 2 tracks correctly.
                        if args.warmup_snapshot and not warmup_locked:
                            global_frame_idx = chunk_start + j
                            is_clear = _frame_conf >= args.warmup_clarity_confidence
                            if is_clear:
                                warmup_clear_seen += 1
                                held = sam2_conf_map >= args.warmup_sam2_confidence
                                warmup_hold_ctr = np.where(held, warmup_hold_ctr + 1, 0)
                                newly = ((warmup_hold_ctr >= args.warmup_persist_frames)
                                         & ~warmup_candidate)
                                growth = int(newly.sum())
                                warmup_candidate |= newly

                                # Smooth growth over a short window; track its peak.
                                warmup_growth_win.append(growth)
                                if len(warmup_growth_win) > 3:
                                    warmup_growth_win.pop(0)
                                smoothed = sum(warmup_growth_win) / len(warmup_growth_win)
                                warmup_peak_growth = max(warmup_peak_growth, smoothed)

                                # Lock on inflection: growth has fallen well below its peak
                                # after enough clear frames and a non-trivial region formed.
                                inflected = (
                                    warmup_clear_seen >= args.warmup_min_frames
                                    and warmup_peak_growth > 0
                                    and smoothed < args.warmup_inflection_ratio * warmup_peak_growth
                                )
                                if inflected:
                                    warmup_snapshot = warmup_candidate.copy()
                                    warmup_locked = True
                                    warmup_lock_frame = global_frame_idx
                                    warmup_lock_reason = 'inflection'
                                    print(f"    [warmup] Snapshot locked at frame "
                                          f"{global_frame_idx + 1} (inflection): "
                                          f"{int(warmup_snapshot.sum())} px")

                            # Fallback cap: lock whatever we have if we never inflect.
                            if (not warmup_locked
                                    and global_frame_idx + 1 >= args.warmup_max_frames):
                                warmup_snapshot = warmup_candidate.copy()
                                warmup_locked = True
                                warmup_lock_frame = global_frame_idx
                                warmup_lock_reason = 'max-frames'
                                print(f"    [warmup] Snapshot locked at frame "
                                      f"{global_frame_idx + 1} (max-frames fallback): "
                                      f"{int(warmup_snapshot.sum())} px")

                        protected = (validated_union | persistent_union | warmup_snapshot) & ~suspended

                        # Symmetric removal: where the model is confidently BACKGROUND
                        # but the mask (SAM 2, or a prior blend) says foreground, drop
                        # those pixels. Without this, blending is one-directional -- it
                        # can only rescue false negatives, never correct false positives,
                        # so any SAM 2 drift or smoke-induced misdetection is permanent
                        # once it appears (confirmed as the cause of both slow static-scene
                        # drift and sudden smoke-triggered grasping onto wrong features).
                        if args.blend_confidence_low is not None:
                            below = conf_for_thresh < args.blend_confidence_low
                            low_conf_counter = np.where(below, low_conf_counter + 1, 0)
                            stale_red = (remaining_mask
                                        & (low_conf_counter >= args.removal_hysteresis)
                                        & ~protected)
                            n_stale = int(stale_red.sum())
                            if n_stale > 0:
                                n_blend_removed_px += n_stale
                                n_blend_removed_frames += 1
                            remaining_mask = remaining_mask & ~stale_red

                    # Gap fill: dilate red into adjacent background pixels
                    if args.fill_gap and args.fill_gap_kernel > 0:
                        import cv2
                        k = args.fill_gap_kernel | 1
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                        rem_d = cv2.dilate(remaining_mask.astype(np.uint8), kernel).astype(bool)
                        bg = ~remaining_mask
                        remaining_mask = remaining_mask | (rem_d & bg)

                    # Spatial constraint: suppress red outside epoch-60 tumor region
                    if spatial_engine is not None:
                        _sp_img = np.array(Image.open(fp).convert('RGB'))
                        spatial_mask = get_spatial_mask(
                            spatial_engine, _sp_img, H_orig, W_orig, args.spatial_dilate
                        )
                        remaining_mask = remaining_mask & spatial_mask

                    # Lateral-growth clip: block the mask from creeping sideways into
                    # tumor the toolhead has not swept. The allowed zone is the
                    # accumulated swept baseline plus a vertical slab around the
                    # toolhead's current horizontal position (top band of the model
                    # prediction). New area outside it is lateral drift -> clipped.
                    # Never removes baseline pixels, so already-swept mask is untouched.
                    if args.limit_lateral_growth and args.blend_epoch61:
                        model_region = conf_for_thresh >= args.threshold
                        ys, xs = np.where(model_region)
                        slab = np.zeros((H_orig, W_orig), dtype=bool)
                        if xs.size > 0:
                            y_top = ys.min()
                            band = ys <= (y_top + args.growth_frontier_band)
                            if band.any():
                                x_lo = max(0, int(xs[band].min()) - args.growth_frontier_margin)
                                x_hi = min(W_orig - 1, int(xs[band].max()) + args.growth_frontier_margin)
                                slab[:, x_lo:x_hi + 1] = True
                        allowed_zone = growth_baseline | slab
                        clipped = remaining_mask & ~allowed_zone
                        n_clip = int(clipped.sum())
                        if n_clip > 0:
                            n_lateral_clipped_px += n_clip
                            n_lateral_clipped_frames += 1
                        remaining_mask = remaining_mask & allowed_zone
                        growth_baseline |= remaining_mask

                    # DISPLAY-ONLY re-assert: fill accumulated protected pixels into the
                    # SAVED mask so SAM 2 shrinking its tracked region mid-chunk can't
                    # make already-accumulated mask vanish -- WITHOUT feeding them into
                    # the SAM 2 carry/seed (that would re-assert the region into SAM 2
                    # and defeat revoke-based arc retraction). carry_source keeps the
                    # true tracked mask for the carry; remaining_mask is display only.
                    carry_source = remaining_mask
                    if args.reassert_protected and args.blend_epoch61:
                        remaining_mask = remaining_mask | protected

                    # Rolling buffer for majority-vote / confidence-weighted carry
                    # (uses carry_source = pre-reassert tracked mask, not the display).
                    if args.blend_epoch61 and args.blend_carry_frames > 1:
                        blended_buffer_rem.append(carry_source.copy())
                        blended_buffer_weights.append(_frame_conf)
                        if len(blended_buffer_rem) > args.blend_carry_frames:
                            blended_buffer_rem.pop(0)
                            blended_buffer_weights.pop(0)

                    if j == len(chunk_paths) - 1:
                        blended_last_remaining = carry_source.copy()

                    if not remaining_mask.any():
                        n_empty += 1

                    if green_engine is not None:
                        _green_img = np.array(Image.open(fp).convert('RGB'))
                        _g_probs, _ = green_engine.predict_single(_green_img)
                        import torch.nn.functional as F
                        _g_conf = F.interpolate(
                            _g_probs[1].unsqueeze(0).unsqueeze(0),
                            size=(H_orig, W_orig),
                            mode='bilinear', align_corners=False,
                        ).squeeze().cpu().numpy()
                        green_mask = _g_conf >= args.green_confidence
                    else:
                        green_mask = seg.get(3, np.zeros((H_orig, W_orig), dtype=bool))

                    rgb_mask     = compose_rgb_mask(remaining_mask, H_orig, W_orig, green=green_mask, color=mask_color)
                    original_img = np.array(Image.open(fp).convert('RGB'))
                    stem         = fp.stem

                    Image.fromarray(rgb_mask.astype(np.uint8)).save(
                        output_dir / f'{stem}_mask.png'
                    )
                    Image.open(fp).save(output_dir / f'{stem}_org.png')

                    if viz_generator is not None:
                        overlay = viz_generator.generate_overlay(
                            original_img, rgb_mask, alpha=args.alpha
                        )
                        Image.fromarray(overlay.astype(np.uint8)).save(
                            output_dir / f'{stem}_overlay.png'
                        )

                    frames_written += 1

                # Feed blended masks back as carry for next chunk
                if args.blend_epoch61 and blended_last_remaining is not None:
                    if args.blend_carry_frames > 1 and len(blended_buffer_rem) > 1:
                        rem_stack = np.stack(blended_buffer_rem, axis=0)
                        if args.blend_carry_weighted:
                            weights = np.array(blended_buffer_weights, dtype=np.float32)
                            w_sum = weights.sum()
                            if w_sum > 0:
                                weights /= w_sum
                            else:
                                weights = np.ones(len(weights)) / len(weights)
                            carry_remaining = (weights[:, None, None] * rem_stack).sum(axis=0) > 0.5
                        else:
                            n = len(blended_buffer_rem)
                            carry_remaining = rem_stack.sum(axis=0) > (n / 2)
                    else:
                        carry_remaining = blended_last_remaining

                    # Reinit: replace carry with fresh model prediction every N frames
                    if args.reinit_every > 0 and chunk_end % args.reinit_every < chunk_size:
                        import torch.nn.functional as F
                        reinit_fp = chunk_paths[-1]
                        reinit_img = np.array(Image.open(reinit_fp).convert('RGB'))
                        reinit_probs, _ = engine.predict_single(reinit_img)
                        reinit_conf = F.interpolate(
                            reinit_probs[1].unsqueeze(0).unsqueeze(0),
                            size=(H_orig, W_orig),
                            mode='bilinear', align_corners=False,
                        ).squeeze().cpu().numpy()
                        carry_remaining = reinit_conf >= args.threshold
                        print(f"    [reinit] Re-seeded SAM 2 from model at frame {chunk_end}")

                    # Restore protected regions into the next chunk's seed: guards against
                    # SAM 2 outright losing an already-validated region (e.g. a smoke-out
                    # or occlusion causing a whole chunk's tracking to fail). Distinct from
                    # the removal-side exemptions above, which only ever protect pixels
                    # SAM 2 currently has as foreground -- they do nothing once SAM 2 has
                    # already dropped a pixel without going through blend/carry-anchor-low.
                    # Runs before --carry-anchor so both restoration paths compose.
                    if args.carry_restore_protected:
                        protected_now = (validated_union | persistent_union | warmup_snapshot) & ~suspended
                        if args.revoke_agreement:
                            # Don't re-inject pixels mid-revocation. revoke_counter > 0
                            # means model + SAM 2 agree background in a clear scene right
                            # now (genuine retraction in progress) -- re-seeding SAM 2 with
                            # them would make SAM 2 re-assert them and prevent the revoke
                            # from ever completing (the "mask stayed there" deadlock).
                            # During smoke the counter is 0, so restoration still fires.
                            protected_now = protected_now & (revoke_counter == 0)
                        restored = protected_now & ~carry_remaining
                        n_restored = int(restored.sum())
                        if n_restored > 0:
                            n_carry_restored_px += n_restored
                            n_carry_restored_frames += 1
                        carry_remaining = carry_remaining | protected_now

                    # Carry anchor: restore pixels the red model is very confident about
                    if args.carry_anchor:
                        import torch.nn.functional as F
                        last_fp = chunk_paths[-1]
                        anchor_img = np.array(Image.open(last_fp).convert('RGB'))
                        anchor_probs, _ = engine.predict_single(anchor_img)  # (2, H, W)

                        anchor_up = F.interpolate(
                            anchor_probs.unsqueeze(0),
                            size=(H_orig, W_orig),
                            mode='bilinear', align_corners=False,
                        ).squeeze(0).cpu().numpy()  # (2, H, W)

                        e61_rem_prob = anchor_up[1]

                        # Restoration: add pixels model is very confident about,
                        # regardless of majority vote — recovers regions SAM 2 dropped
                        carry_remaining = carry_remaining | (e61_rem_prob > args.carry_anchor_high)

                        # Symmetric removal: prune carry pixels the model is confidently
                        # background about, before they get handed to the next chunk as
                        # SAM 2's seed. Same one-directional gap as the per-frame blend,
                        # but here it protects the seed itself -- an uncorrected seed
                        # corrupts every frame in the next chunk, not just one.
                        if args.carry_anchor_low is not None:
                            stale_carry = (carry_remaining
                                          & (e61_rem_prob < args.carry_anchor_low)
                                          & ~(validated_union | persistent_union))
                            n_stale_carry = int(stale_carry.sum())
                            if n_stale_carry > 0:
                                n_carry_removed_px += n_stale_carry
                                n_carry_removed_frames += 1
                            carry_remaining = carry_remaining & ~stale_carry

            print(f"    → {frames_written}/{len(frame_paths)} frames written")

        # ---- Summary ---------------------------------------------------------
        total_elapsed = time.perf_counter() - t_total_start
        fps = len(frame_paths) / total_elapsed

        print()
        print("=" * 70)
        print("Inference complete")
        print("=" * 70)
        print(f"Frames processed:        {len(frame_paths)}")
        print(f"Frames with empty mask:  {n_empty}")
        if args.blend_confidence_low is not None:
            print(f"Symmetric blend removal (per-frame, threshold={args.blend_confidence_low}): "
                  f"fired on {n_blend_removed_frames}/{len(frame_paths)} frames, "
                  f"{n_blend_removed_px} px total removed")
        if args.carry_anchor_low is not None:
            print(f"Symmetric carry-anchor removal (chunk boundary, threshold={args.carry_anchor_low}): "
                  f"fired on {n_carry_removed_frames} boundaries, "
                  f"{n_carry_removed_px} px total removed")
        if args.protect_confidence is not None:
            print(f"Permanent protection map (threshold={args.protect_confidence}): "
                  f"{int(validated_union.sum())} px protected by end of episode "
                  f"({100*validated_union.sum()/validated_union.size:.1f}% of frame)")
        if args.sam2_persistence_frames is not None:
            print(f"SAM2-persistence protection (>={args.sam2_persistence_frames} consecutive "
                  f"frames, earned within first {args.sam2_persistence_window} frames): "
                  f"{int(persistent_union.sum())} px protected "
                  f"({100*persistent_union.sum()/persistent_union.size:.1f}% of frame)")
        if args.carry_restore_protected:
            print(f"Carry-restore-protected: fired on {n_carry_restored_frames} boundaries, "
                  f"{n_carry_restored_px} px total re-injected into next chunk's seed")
        if args.protect_revoke_frames is not None:
            _mode = ("model+SAM2 agree bg, clear scene" if args.revoke_agreement
                     else "model-only")
            print(f"Protection revocation ({_mode}, mode={args.revoke_mode}, "
                  f">={args.protect_revoke_frames} consecutive frames): "
                  f"{n_protection_revoked_px} px total revoked "
                  f"(cumulative, may include pixels revoked more than once)")
            if args.revoke_mode == 'suspend':
                print(f"  Suspended (recoverable) at episode end: {int(suspended.sum())} px")
        if args.limit_lateral_growth:
            print(f"Lateral-growth clip (margin={args.growth_frontier_margin}px): "
                  f"fired on {n_lateral_clipped_frames}/{len(frame_paths)} frames, "
                  f"{n_lateral_clipped_px} px total clipped as lateral drift")
        if args.warmup_snapshot:
            if warmup_locked:
                print(f"Warm-up snapshot: locked at frame {warmup_lock_frame + 1} "
                      f"({warmup_lock_reason}), {int(warmup_snapshot.sum())} px "
                      f"({100*warmup_snapshot.sum()/warmup_snapshot.size:.1f}% of frame)")
            else:
                print("Warm-up snapshot: never locked (episode ended before "
                      "establishment inflection or fallback cap)")
        print(f"Total time (s):          {total_elapsed:.1f}")
        print(f"Effective FPS:           {fps:.1f}")
        print(f"Output directory:        {output_dir}")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
