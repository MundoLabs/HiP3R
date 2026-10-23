<div align="center">

<p><img src="assets/logo.svg" alt="HiP3R" height="100"></p>
<h3>Hierarchical Pose Graphs for Long-Horizon 3D Reconstruction</h3>

<p>
  <a href="https://kkakkkka.github.io/HiP3R/">
    <img src="https://img.shields.io/badge/Project-Page-blue" alt="Project page">
  </a>
  <a href="https://arxiv.org/abs/2610.31635">
    <img src="https://img.shields.io/badge/arXiv-Abs-b31b1b?logo=arxiv" alt="arXiv Abs">
  </a>
  <a href="https://arxiv.org/pdf/2610.31635">
    <img src="https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader" alt="Paper PDF">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="License">
  </a>
</p>

<p>
  Reconstruct long image sequences by aligning overlapping chunks and refining camera poses.<br>
  Inference with VGGT-Omega or LingBot-MAP, with SALAD loop retrieval.
</p>

<p>
  <a href="#demo">Demo</a> &nbsp;·&nbsp;
  <a href="#installation">Installation</a> &nbsp;·&nbsp;
  <a href="#weights">Weights</a> &nbsp;·&nbsp;
  <a href="#input">Input</a> &nbsp;·&nbsp;
  <a href="#inference">Inference</a> &nbsp;·&nbsp;
  <a href="#outputs">Outputs</a>
</p>

</div>

---

## Demo

<div align="center">

https://github.com/user-attachments/assets/8db6798e-9e9c-44fa-9354-e66d826542a9

</div>

## Installation

Use Python 3.10 or newer. The demo defaults to an NVIDIA GPU with CUDA and
bfloat16 inference.

From the repository root:

```bash
conda create -n hip3r python=3.10
conda activate hip3r
pip install -e '.[fast]'
```

Check that PyTorch can access your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Weights

Download the checkpoints from the corresponding upstream projects and place
them in `checkpoints/`.

| Model | Checkpoint | Source |
| --- | --- | --- |
| VGGT-Omega | `vggt_omega_1b_512.pt` | [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) |
| LingBot-MAP | `lingbot-map-long.pt` | [LingBot-MAP](https://github.com/Robbyant/lingbot-map) |
| SALAD | `dino_salad.ckpt` | [SALAD](https://github.com/serizba/salad) |
| DINOv2 ViT-B/14 | `dinov2_vitb14_pretrain.pth` | [DINOv2](https://github.com/facebookresearch/dinov2) |

```text
checkpoints/
├── vggt_omega_1b_512.pt
├── lingbot-map-long.pt
├── dino_salad.ckpt
└── dinov2_vitb14_pretrain.pth
```

Download the reconstruction checkpoint for your chosen backend, plus both
`dino_salad.ckpt` and the DINOv2 checkpoint for the required SALAD loop retrieval.

## Input

The demo accepts a video or a directory of JPG/PNG frames. For an image
sequence, name the files so that alphabetical order matches frame order:

```text
data/example/
├── 000000.png
├── 000001.png
├── 000002.png
└── ...
```

Video input is decoded into frames under `OUTPUT/input_frames/`.
Use a fresh output directory for each video run.

## Inference

Edit the settings in [run.sh](run.sh). Each `${VAR:-value}` expression supplies
a default that you can also override through an environment variable.

```bash
INPUT="${INPUT:-data/example}" # Or data/example.mp4.
OUTPUT="${OUTPUT:-outputs/example}"
BACKEND="${BACKEND:-vggt_omega}"
CHECKPOINT="${CHECKPOINT:-checkpoints/vggt_omega_1b_512.pt}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
```

Run:

```bash
bash run.sh
```

To use LingBot-MAP without editing the script:

```bash
BACKEND=lingbot_map CHECKPOINT=checkpoints/lingbot-map-long.pt bash run.sh
```

You can also pass arguments directly to Python:

```bash
python demo.py \
  --input data/example \
  --output outputs/example \
  --backend vggt_omega \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --device cuda:0 \
  --dtype bfloat16
```

Use `python demo.py --help` for all options. Relative paths in `run.sh` are
resolved from the repository root.

VGGT-Omega uses 60-frame chunks with 30-frame overlap. LingBot-MAP uses
120-frame chunks with 32-frame overlap. Both use keyframe pose refinement.
SALAD loop retrieval runs before reconstruction for both backends.

## Outputs

Results are saved to `OUTPUT`:

| File | Contents |
| --- | --- |
| `camera_poses.txt` | One flattened 4×4 camera-to-world matrix per frame |
| `intrinsics.txt` | One flattened 3×3 camera intrinsic matrix per frame |
| `chunks/chunk_*.npz` | Per-chunk points, confidence, poses, intrinsics, and available depth fields |
| `reconstruction.json` | Chunk ranges, alignment transforms, and reconstruction diagnostics |

Load the camera poses and chunk geometry with NumPy:

```python
import numpy as np

poses = np.loadtxt("outputs/example/camera_poses.txt").reshape(-1, 4, 4)
with np.load("outputs/example/chunks/chunk_0000.npz") as chunk:
    points = chunk["world_points"]
    confidence = chunk["world_points_conf"]
```

## Acknowledgements

This code uses [VGGT-Omega](https://github.com/facebookresearch/vggt-omega),
[LingBot-MAP](https://github.com/Robbyant/lingbot-map),
[SALAD](https://github.com/serizba/salad), and
[DINOv2](https://github.com/facebookresearch/dinov2). Third-party notices are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Original HiP3R code is released under the [Apache License 2.0](LICENSE).
Bundled third-party code and model weights remain subject to their respective
licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
