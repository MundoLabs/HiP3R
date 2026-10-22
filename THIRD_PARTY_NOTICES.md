# Third-party notices

The Apache 2.0 license for original HiP3R code does not replace the licenses
of bundled third-party code or model weights.

This repository includes inference code for the bundled backends.
Optional plain VGGT and ABot-Recon backends use external source checkouts.

- `src/vggt_omega/` is derived from VGGT/VGGT-Omega. Its notices are
  retained in `licenses/VGGT_LICENSE.txt` and `licenses/VGGT_OMEGA_LICENSE.txt`.
- `src/lingbot_map/` is derived from LingBot-MAP (Apache-2.0). Its
  license is retained in `licenses/LINGBOT_MAP_LICENSE.txt`.
- `src/dinov2/` contains the DINOv2 inference subset used
  by SALAD. Its Apache-2.0 license is retained in `licenses/DINOV2_LICENSE.txt`.
- `src/loop_models/aggregators/salad.py` contains code adapted from
  OpenGlue as stated in its source header.

No model checkpoint is included in this repository.
