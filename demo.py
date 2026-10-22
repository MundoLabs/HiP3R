"""Reconstruct a video or an ordered image directory."""

import argparse
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Video or JPG/PNG directory")
    parser.add_argument("--output", type=Path, default=Path("outputs/example"))
    parser.add_argument("--backend", choices=["vggt_omega", "lingbot_map"], default="vggt_omega")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--salad-checkpoint", type=Path, default=Path("checkpoints/dino_salad.ckpt"))
    parser.add_argument("--dinov2-checkpoint", type=Path,
                        default=Path("checkpoints/dinov2_vitb14_pretrain.pth"))
    return parser.parse_args(argv)


def extract_video_frames(video: Path, output: Path) -> Path:
    import cv2

    frame_root = output / "input_frames"
    # A fresh directory prevents stale frames from entering a reconstruction.
    frame_root.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(video))
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            target = frame_root / f"{count:06d}.png"
            if not cv2.imwrite(str(target), frame):
                raise OSError(f"failed to write frame: {target}")
            count += 1
    finally:
        capture.release()
    if count == 0:
        raise ValueError(f"video contains no decodable frames: {video}")
    return frame_root


def main(argv=None):
    args = parse_args(argv)

    from long_reconstruction.backends import create_local_backend
    from long_reconstruction.loop_retrieval import SaladRetrievalConfig, retrieve_salad_loop_pairs
    from long_reconstruction.method import method_config
    from long_reconstruction.runner import ReconstructionRunner, generate_chunk_ranges

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    weights = [args.checkpoint, args.salad_checkpoint, args.dinov2_checkpoint]
    for path in weights:
        if not path.is_file():
            raise FileNotFoundError(path)

    image_dir = args.input if args.input.is_dir() else extract_video_frames(args.input, args.output)
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError(f"no JPG or PNG frames found in {image_dir}")

    chunk_size, overlap = (120, 32) if args.backend == "lingbot_map" else (60, 30)
    config = method_config(
        "keyframe",
        chunk_size=chunk_size,
        overlap=overlap,
        save_chunk_predictions=True,
        hierarchical_max_nfev=100,
        hierarchical_coarse_max_nfev=5000,
        hierarchical_coarse_adaptive_budget_fallback_max_nfev=5000,
        hierarchical_require_solver_convergence=True,
    )
    ranges = generate_chunk_ranges(len(images), chunk_size, overlap)
    loop_pairs, _ = retrieve_salad_loop_pairs(
        image_dir,
        ranges,
        SaladRetrievalConfig(
            salad_checkpoint=args.salad_checkpoint,
            dinov2_checkpoint=args.dinov2_checkpoint,
        ),
        args.output / "loop_retrieval",
    )

    options = {}
    if args.backend == "lingbot_map":
        options.update(
            mode="windowed",
            pose_convention="decoded_c2w",
            geometry_source="depth_unprojection",
            window_size=chunk_size,
            overlap_size=overlap,
            use_sdpa=True,
        )
    adapter, _ = create_local_backend(
        args.backend, checkpoint=args.checkpoint, device=args.device,
        inference_dtype=args.dtype, **options,
    )
    runner = ReconstructionRunner(adapter, config, progress=print)
    result = runner.run(images, loop_pairs=loop_pairs, output_dir=args.output)
    print(f"Reconstructed {len(images)} frames. Results saved to {args.output}")
    return result


if __name__ == "__main__":
    main()
