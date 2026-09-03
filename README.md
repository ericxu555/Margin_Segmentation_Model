# Tumor Margin Segmentation Pipeline

Segmentation of the remaining uncharred tissue margin from 2D endoscope video.
The pipeline runs two models per frame: a full-tumor segmentation model that
constrains where the margin can appear, and a margin model that segments the
margin itself within that region.

The margin is the band of viable tissue left between where the tool has cut and
where the tumor ends, so it is what indicates whether a resection has gone deep
enough without going too far. It is also difficult to segment: it occupies
roughly 2 percent of the pixels in a frame, so a model that predicts nothing at
all still scores 98 percent pixel accuracy.

## Pipeline overview

1. **Full-tumor segmentation (Stage 1).** A U-Net with a ResNet-34 encoder
   segments the entire tumor in every frame. Its output is dilated by 20 pixels
   and used as a spatial constraint, so the margin can only be reported on or
   near the tumor.
2. **Margin segmentation (Stage 2).** A second U-Net with the same architecture
   segments the uncharred margin, thresholded at 0.6 confidence.
3. **Temporal smoothing and gating.** The margin confidence is smoothed across
   frames with an exponential moving average, then ANDed with the Stage 1
   constraint.
4. **Video rendering.** The per-frame overlays are assembled into a finished
   video, named after the input episode.

SAM2 is deliberately not used. It was tested on this target and drifts, so
inference is per-frame.

## Setup

Run every command from the repository root with the virtual environment
activated. A CUDA-capable GPU is recommended; the pipeline falls back to CPU
automatically but runs considerably slower.

### Step 1. Clone the repository and create an environment

```bash
git clone https://github.com/ericxu555/Margin_Segmentation_Model.git
cd Margin_Segmentation_Model

python -m venv venv
# Windows:       venv\Scripts\activate
# macOS/Linux:   source venv/bin/activate
```

### Step 2. Install PyTorch

Install PyTorch first and separately, because the CUDA builds are not
distributed on PyPI and will not resolve from a plain `pip install`. Use the
selector at https://pytorch.org/get-started/locally/ to get the command for
your CUDA version. For CUDA 12.8, which this pipeline was developed against:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

For CPU only:

```bash
pip install torch torchvision
```

Verify before continuing:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Step 3. Install the remaining Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4. Download the model weights

Two checkpoints are required, roughly 98 MB each. Neither is included in this
repository because both exceed GitHub's file size limit.

**4a. Margin segmentation checkpoint (Stage 2).** Download both files and place
them in `checkpoints_margin58/`:

> https://drive.google.com/drive/folders/1bBXVkgTYPiGZaNKX1Yo6_3pTL2gWmGtP

```
checkpoints_margin58/
  best_model_epoch_swa.pth
  best_model_epoch_swa_metadata.json
```

**4b. Full-tumor segmentation checkpoint (Stage 1).** This is the same model
used by the toolhead-following pipeline, from the same folder:

> https://drive.google.com/drive/folders/18w_LTyrIgShmxdWPbZUZF8BunnrfqyQo

```
checkpoints_tumor_binary10/
  best_model_epoch_swa.pth
  best_model_epoch_swa_metadata.json
```

The `_metadata.json` files are already in this repository, so only the two
`.pth` files strictly need downloading. Do not replace the metadata: it tells
the loader which architecture to build and is checksum-verified against the
weights, so a mismatched pair fails at load time rather than producing wrong
output silently.

### Step 5. Run the pipeline

```bash
python run_margin.py <video_frame_dir> <output_dir>
```

See Usage below for the arguments and outputs.

## Configuration

If you keep the weights outside the repository, point at them with environment
variables instead of editing any script. All four are optional and have working
defaults.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MARGIN_MODEL` | `checkpoints_margin58/best_model_epoch_swa.pth` | Margin weights (Stage 2) |
| `MARGIN_META` | `checkpoints_margin58/best_model_epoch_swa_metadata.json` | Margin metadata |
| `SPATIAL_MODEL` | `checkpoints_tumor_binary10/best_model_epoch_swa.pth` | Tumor weights (Stage 1) |
| `SPATIAL_META` | `checkpoints_tumor_binary10/best_model_epoch_swa_metadata.json` | Tumor metadata |

```bash
export MARGIN_MODEL=/path/to/best_model_epoch_swa.pth
export MARGIN_META=/path/to/best_model_epoch_swa_metadata.json
```

## Model performance

| | Stage 1: full tumor | Stage 2: margin |
| --- | --- | --- |
| Checkpoint | `binary10` SWA | `margin58` SWA |
| Architecture | U-Net / ResNet-34 | U-Net / ResNet-34 |
| Input | 512x512, 2-class | 512x512, 2-class |
| Training frames | 497 train / 61 val | 345 train / 86 val |
| Target pixel fraction | 14.8% | 2.09% |
| Validation Dice | 0.976 | 0.842 |
| Out-of-distribution Dice | 0.934 | not measured |

Both checkpoints are SWA averages, Stage 1 from epoch 50 and Stage 2 from
epoch 35, each of which outperformed every individual epoch.

Stage 2 was trained with a compound loss of Focal (gamma=2.75), Tversky
(alpha=0.10, beta=0.90), and Boundary (sigma=3.0). The Tversky term is strongly
asymmetric, penalizing false negatives nine times more heavily than false
positives, because the target is thin and rare and missing it is the costlier
error. The training set includes 50 hard-negative frames of unresected tumor
with deliberately empty labels, which teach the model when not to segment.

## Usage

### Input data

The pipeline takes **a folder containing the frames of one resection episode**,
as individual PNG files. It does not accept a video file.

```
my_case/
  frame000001.png
  frame000002.png
  frame000003.png
  ...
```

Three requirements:

1. **Filenames must sort into playback order.** Frames are read with a plain
   alphabetical sort, so zero-padded names such as `frame000001.png` are
   required. Unpadded names break the ordering, because `frame10.png` sorts
   before `frame2.png`.
2. **One episode per folder**, with no unrelated images mixed in. Files ending
   in `_mask`, `_org`, or `_overlay` are skipped, so an output directory can
   safely be reused as input.
3. **PNG is the tested format.** JPEG, BMP, and TIFF are also accepted.

Any frame size works; the models resize internally. The example videos are
540x540.

To convert an existing video file into this layout:

```bash
ffmpeg -i my_case.mp4 my_case/frame%06d.png
```

### Running the pipeline

```bash
python run_margin.py <video_frame_dir> <output_dir> [options]
```

For example:

```bash
python run_margin.py \
    /data/Ablation/full_balanced_exp1/20260602-001011-226627/endoscope \
    /results/case_226627
```

### Pipeline output

For each input frame, three images are written to the output directory:

| File | Contents |
| --- | --- |
| `<frame>_overlay.png` | Original frame with the margin drawn over it in red. Check this first. |
| `<frame>_mask.png` | The margin mask alone. This is the file to use for quantitative analysis. |
| `<frame>_org.png` | A copy of the input frame, so each output is self-contained. |

The overlay frames are then rendered into a finished video, written into the
same directory and named after the **input episode** rather than the output
directory, so the file identifies its source:

```
/data/Ablation/full_balanced_exp1/20260602-001011-226627/endoscope
  ->  <output_dir>/full_balanced_exp1_20260602-001011-226627.mp4
```

The name is taken from the two directories above the frames when they sit in a
generic container such as `endoscope`; otherwise the frames directory's own
name is used.

### Options

| Option | Default | Effect |
| --- | --- | --- |
| `--threshold` | `0.6` | Confidence a pixel needs to be called margin. Lower to 0.4 or 0.5 on low-contrast or unusually dark video. |
| `--fps` | `30` | Frame rate of the rendered video. |
| `--no-video` | off | Write the frames but skip rendering the video. |
| `--device` | `cuda` | Use `cpu` if no GPU is available. Results are identical, only slower. |

```bash
python run_margin.py <frames> <out> --threshold 0.5
python run_margin.py <frames> <out> --fps 10
python run_margin.py <frames> <out> --no-video
```

`--fps` and `--no-video` are handled by `run_margin.py`. Every other option is
passed through to `infer_sam2_red.py`, and because argparse takes the last
occurrence of a repeated flag, anything you append overrides the default.

### The command it runs

`run_margin.py` is a convenience wrapper that fills in the validated settings.
The exact command it executes is:

```bash
python infer_sam2_red.py \
    --no-sam2 \
    --model checkpoints_margin58/best_model_epoch_swa.pth \
    --metadata checkpoints_margin58/best_model_epoch_swa_metadata.json \
    --input <video_frame_dir> \
    --output-dir <output_dir> \
    --threshold 0.6 \
    --blend-ema-alpha 0.3 \
    --spatial-model checkpoints_tumor_binary10/best_model_epoch_swa.pth \
    --spatial-metadata checkpoints_tumor_binary10/best_model_epoch_swa_metadata.json \
    --spatial-dilate 20 \
    --device cuda \
    --visualize
```

Running that directly gives an identical result, minus the video.

| Setting | Value | Purpose |
| --- | --- | --- |
| `--no-sam2` | on | SAM2 drifts on this target, so inference is per-frame |
| `--threshold` | 0.6 | Confidence needed to call a pixel margin |
| `--blend-ema-alpha` | 0.3 | EMA temporal smoothing of the confidence map, which suppresses flicker. `1.0` disables it |
| `--spatial-dilate` | 20 | Dilation of the Stage 1 tumor mask before gating, so the margin at the tumor edge is not clipped |

### Rendering a video separately

To build a video from frames you already have:

```bash
python frames_to_video.py \
    --input <output_dir> \
    --output <output_dir>/episode.mp4 \
    --fps 30
```

### Runtime

On a CUDA GPU, inference runs at roughly 8 frames per second, so a 962 frame
episode takes about two minutes plus a few seconds to render the video.
CPU-only runs are substantially slower.

### Advanced use

`infer_sam2_red.py` exposes considerably more control than the wrapper,
including test-time augmentation, precomputed spatial masks, Gaussian
smoothing, and frame ranges:

```bash
python infer_sam2_red.py --help
```

## Troubleshooting

**`No matching distribution found for torch==...+cu128`.** The CUDA builds of
PyTorch are not on PyPI. Install PyTorch separately first, as described in
Step 2, then run `pip install -r requirements.txt`.

**Missing model weights.** `run_margin.py` checks for all four files before
starting and names any that are absent. Confirm the two `.pth` files are in the
directories listed in Step 4, or set the environment variables in
Configuration.

**Checksum mismatch on load.** The `.pth` and its `_metadata.json` must come
from the same checkpoint. Re-download rather than mixing files.

**`torch.cuda.is_available()` returns `False`.** The pipeline runs on CPU but
considerably slower. Reinstall PyTorch using the index URL matching your CUDA
version, as described in Step 2.

**CUDA out of memory.** Add `--device cpu`. Results are identical, only slower.

**Nothing is segmented.** If masks are empty across a whole episode, the
footage may be lower contrast than the training data. Try `--threshold 0.4`. If
the tumor itself is not being found, Stage 1 is the problem rather than
Stage 2; `--no-spatial-constraint` confirms this by disabling the gate.

**The margin is clipped at the tumor edge.** Increase `--spatial-dilate` to 40.
The margin sits directly at the tumor boundary, so a constraint that is too
tight clips the structure being segmented.

## Repository layout

```
src/                          segmentation model and inference engine
  model.py                    U-Net/ResNet-34 architecture definition
  inference_engine.py         loads checkpoint, runs per-frame inference
  preprocessor.py / postprocessor.py
  visualization.py            overlay rendering
  model_persistence.py        checkpoint loading and checksum verification
run_margin.py                 end-to-end driver: both stages, then the video
infer_sam2_red.py             inference script the wrapper calls
frames_to_video.py            renders overlay frames into an .mp4
checkpoints_margin58/         margin weights (Stage 2)
checkpoints_tumor_binary10/   full-tumor weights (Stage 1)
requirements.txt
```
