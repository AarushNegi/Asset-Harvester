"""
batch_inference.py
------------------
Runs Asset Harvester inference on all formatted samples in inputs/samples/
and saves one gaussians.ply per object to outputs/.

Usage (local with good GPU):
    python scripts/batch_inference.py

Usage (Kaggle T4 / limited GPU):
    python scripts/batch_inference.py --fp16 --offload --max_samples 5

Usage (single sample test):
    python scripts/batch_inference.py --max_samples 1
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path


# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_SAMPLES_DIR    = "inputs/samples"
DEFAULT_OUTPUT_DIR     = "outputs"
DEFAULT_NVIDIA_REPO    = "nvidia-asset-harvester"
DEFAULT_CHECKPOINTS    = "checkpoints"

DIFFUSION_CHECKPOINT   = "AH_multiview_diffusion.safetensors"
LIFTING_CHECKPOINT     = "AH_tokengs_lifting.safetensors"
CAMERA_CHECKPOINT      = "AH_camera_estimator.safetensors"


# ─── Argument Parser ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference with Asset Harvester")

    parser.add_argument("--samples_dir",   default=DEFAULT_SAMPLES_DIR,
                        help="Folder containing formatted samples (default: inputs/samples)")
    parser.add_argument("--output_dir",    default=DEFAULT_OUTPUT_DIR,
                        help="Where to save outputs (default: outputs)")
    parser.add_argument("--nvidia_repo",   default=DEFAULT_NVIDIA_REPO,
                        help="Path to cloned NVIDIA asset-harvester repo")
    parser.add_argument("--checkpoints",   default=DEFAULT_CHECKPOINTS,
                        help="Path to checkpoints folder")
    parser.add_argument("--max_samples",   type=int, default=None,
                        help="Limit number of samples to process (useful for testing)")
    parser.add_argument("--fp16",          action="store_true",
                        help="Use fp16 precision (required on T4/limited VRAM)")
    parser.add_argument("--offload",       action="store_true",
                        help="Offload model to CPU between steps (required on T4)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip samples that already have gaussians.ply (default: True)")

    return parser.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def check_paths(args):
    """Verify all required paths exist before starting."""
    errors = []

    if not Path(args.samples_dir).exists():
        errors.append(f"Samples dir not found: {args.samples_dir}")

    if not Path(args.nvidia_repo).exists():
        errors.append(f"NVIDIA repo not found: {args.nvidia_repo}")

    ckpt = Path(args.checkpoints)
    for f in [DIFFUSION_CHECKPOINT, LIFTING_CHECKPOINT, CAMERA_CHECKPOINT]:
        if not (ckpt / f).exists():
            errors.append(f"Checkpoint missing: {ckpt / f}")

    if errors:
        print("\n[ERROR] Missing required files:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


def get_all_samples(samples_dir):
    """
    Scan samples_dir for valid Asset Harvester input folders.
    A valid sample has: input_views/frame_00.jpeg + input_views/mask_00.png
    """
    samples = []
    base = Path(samples_dir)

    for sample_folder in sorted(base.iterdir()):
        if not sample_folder.is_dir():
            continue
        input_views = sample_folder / "input_views"
        frame = input_views / "frame_00.jpeg"
        mask  = input_views / "mask_00.png"

        if frame.exists() and mask.exists():
            samples.append(sample_folder)
        else:
            print(f"  [SKIP] {sample_folder.name} — missing frame_00.jpeg or mask_00.png")

    return samples


def build_sample_paths_json(samples, tmp_json_path):
    """
    Build a sample_paths.json in the format Asset Harvester expects.
    Format: list of relative paths from the samples_dir root.
    """
    paths = [str(s.name) for s in samples]
    with open(tmp_json_path, "w") as f:
        json.dump(paths, f, indent=2)
    print(f"  Written sample_paths.json with {len(paths)} entries")


def already_done(output_dir, sample_name):
    """Check if gaussians.ply already exists for this sample."""
    ply = Path(output_dir) / sample_name / "gaussians.ply"
    return ply.exists()


def run_inference(args, samples_dir_abs, output_dir_abs, checkpoints_abs, nvidia_repo_abs, max_samples):
    """Run Asset Harvester run_inference.py as a subprocess."""

    cmd = [
        sys.executable,
        str(nvidia_repo_abs / "run_inference.py"),
        "--data_root",          str(samples_dir_abs),
        "--output_dir",         str(output_dir_abs),
        "--diffusion_checkpoint", str(checkpoints_abs / DIFFUSION_CHECKPOINT),
        "--lifting_checkpoint",   str(checkpoints_abs / LIFTING_CHECKPOINT),
        "--ahc_checkpoint",       str(checkpoints_abs / CAMERA_CHECKPOINT),
    ]

    if args.fp16:
        cmd += ["--precision", "fp16"]

    if args.offload:
        cmd += ["--offload_model_to_cpu"]

    if max_samples is not None:
        cmd += ["--max_samples", str(max_samples)]

    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    print(f"\n[CMD] {' '.join(cmd)}\n")

    result = subprocess.run(cmd, env=env, cwd=str(nvidia_repo_abs))
    return result.returncode


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve all paths to absolute
    samples_dir_abs  = Path(args.samples_dir).resolve()
    output_dir_abs   = Path(args.output_dir).resolve()
    checkpoints_abs  = Path(args.checkpoints).resolve()
    nvidia_repo_abs  = Path(args.nvidia_repo).resolve()

    print("=" * 60)
    print("  Asset Harvester — Batch Inference")
    print("=" * 60)
    print(f"  Samples dir : {samples_dir_abs}")
    print(f"  Output dir  : {output_dir_abs}")
    print(f"  Checkpoints : {checkpoints_abs}")
    print(f"  NVIDIA repo : {nvidia_repo_abs}")
    print(f"  fp16        : {args.fp16}")
    print(f"  offload     : {args.offload}")
    print(f"  max_samples : {args.max_samples}")
    print("=" * 60)

    # Check paths
    check_paths(args)

    # Discover samples
    print(f"\n[1/4] Scanning {samples_dir_abs} for samples...")
    all_samples = get_all_samples(samples_dir_abs)
    print(f"  Found {len(all_samples)} valid samples")

    if not all_samples:
        print("[ERROR] No valid samples found. Run build_sample_input.py first.")
        sys.exit(1)

    # Filter already done
    if args.skip_existing:
        pending = [s for s in all_samples if not already_done(output_dir_abs, s.name)]
        skipped = len(all_samples) - len(pending)
        if skipped:
            print(f"  Skipping {skipped} already completed samples")
    else:
        pending = all_samples

    if not pending:
        print("[DONE] All samples already processed.")
        sys.exit(0)

    # Apply max_samples limit
    if args.max_samples is not None:
        pending = pending[:args.max_samples]
        print(f"  Processing first {len(pending)} samples (--max_samples={args.max_samples})")

    print(f"\n[2/4] {len(pending)} samples to process")

    # Create output dir
    output_dir_abs.mkdir(parents=True, exist_ok=True)

    # Build sample_paths.json inside samples_dir
    tmp_json = samples_dir_abs / "sample_paths.json"
    print(f"\n[3/4] Building sample_paths.json...")
    build_sample_paths_json(pending, tmp_json)

    # Run inference
    print(f"\n[4/4] Running Asset Harvester inference...")
    returncode = run_inference(
        args,
        samples_dir_abs,
        output_dir_abs,
        checkpoints_abs,
        nvidia_repo_abs,
        max_samples=len(pending)
    )

    # Report results
    print("\n" + "=" * 60)
    if returncode == 0:
        print("  Inference completed successfully")
        print(f"\n  Output files:")
        for s in pending:
            ply = output_dir_abs / s.name / "gaussians.ply"
            mp4 = output_dir_abs / s.name / "multiview.mp4"
            ply_status = "✓" if ply.exists() else "✗ MISSING"
            mp4_status = "✓" if mp4.exists() else "✗ MISSING"
            print(f"    {s.name}/")
            print(f"      gaussians.ply  [{ply_status}]")
            print(f"      multiview.mp4  [{mp4_status}]")
    else:
        print(f"  Inference FAILED with return code {returncode}")
        print("  Check the logs above for errors")
    print("=" * 60)


if __name__ == "__main__":
    main()