import os
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Union


# A local copy of utils.utility.Timer: a context manager that reports the
# execution time of the block it wraps.
class Timer:
    """Class for measuring execution time."""
    def __init__(self):
        self.start_time = None

    def start(self):
        """Starts the timer."""
        self.start_time = time.perf_counter()

    def elapsed(self):
        """Returns the elapsed time in seconds since the timer was started."""
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")
        return time.perf_counter() - self.start_time

    def report(self, verbose=True):
        """Ends the timer and report the elapsed time in seconds."""
        e = self.elapsed()
        if verbose:
            print(f"Step executed in {e*1000:.4f} ms ({e:.4f} s)")
        return e

    def __enter__(self):
        """Starts the timer."""
        self.start()
        return self

    def __exit__(self, *args):
        """Ends the timer."""
        self.report(verbose=True)


def validate_path(path: Union[str, Path], name: str) -> None:
    """Validate that the path exists."""
    if path and not Path(path).exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def set_default_paths(
    args: argparse.Namespace,
    defaults: Dict[str, str],
) -> None:
    """Set default paths for the given arguments."""
    for attr, path in defaults.items():
        if not hasattr(args, attr) or getattr(args, attr) is None:
            setattr(args, attr, path)


def run_registration(args: argparse.Namespace) -> None:
    """Run the registration command."""
    command = [
        "unigradicon-register",
        f"--fixed={str(args.fixed_path)}",
        f"--fixed_modality={str(args.fixed_modality)}",
        f"--moving={str(args.moving_path)}",
        f"--moving_modality={str(args.moving_modality)}",
        f"--transform_out={str(args.transform_out)}",
        f"--warped_moving_out={str(args.warped_moving_out)}",
        f"--io_iterations={args.io_iterations}",
        f"--io_sim={args.io_sim}",
        f"--model={args.model}",
    ]

    if args.fixed_segmentation_path:
        command.append(f"--fixed_segmentation={str(args.fixed_segmentation_path)}")
    if args.moving_segmentation_path:
        command.append(f"--moving_segmentation={str(args.moving_segmentation_path)}")

    if args.verbose or args.dry_run:
        print("Running:", " ".join(command))

    if not args.dry_run:
        subprocess.run(command, cwd=args.scratch_dir, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run uniGradICON registration.")

    # CLI arguments
    parser.add_argument(
        "--fixed_path",
        required=True,
        help="Path to the fixed image"
    )

    parser.add_argument(
        "--fixed_modality",
        default="mri",
        help="Modality of the fixed image. Should be 'ct' or 'mri'"
    )

    parser.add_argument(
        "--fixed_segmentation_path",
        help="Path to the segmentation map of the fixed image. "
             "This map will be applied to the fixed image before registration."
    )

    parser.add_argument(
        "--moving_path",
        required=True,
        help="Path to the moving image"
    )

    parser.add_argument(
        "--moving_modality",
        default="mri",
        help="Modality of the moving image. Should be 'ct' or 'mri'"
    )

    parser.add_argument(
        "--moving_segmentation_path",
        help="Path to the segmentation map of the moving image. "
             "This map will be applied to the moving image before registration."
    )

    parser.add_argument(
        "--transform_out",
        help="Path to the output HDF5 transformation file"
    )

    parser.add_argument(
        "--warped_moving_out",
        help="Path to the output warped moving image"
    )

    parser.add_argument(
        "--io_iterations",
        type=int,
        default=50,
        help="Number of instance-optimization (IO) iterations. Default is 50."
    )

    parser.add_argument(
        "--io_sim",
        default="lncc",
        choices=["lncc", "lncc2", "mind"],
        help="Similarity measure used in IO. Default is LNCC. Choose from [lncc, lncc2, mind]."
    )

    parser.add_argument(
        "--model",
        default="unigradicon",
        choices=["unigradicon", "multigradicon"],
        help="The model to load. Default is unigradicon. Choose from [unigradicon, multigradicon]."
    )

    parser.add_argument(
        "--use_mask",
        action="store_true",
        help="Use mask for registration. Default is False."
    )

    parser.add_argument(
        "--suffix",
        help="Suffix to add to the output file names."
    )

    parser.add_argument(
        "--scratch_dir",
        default=str(Path(__file__).resolve().parents[2] / 'assets' / 'registration'),
        help="Path to the scratch directory to store or load the network weights"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set True to print verbose output."
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the command without executing it."
    )

    args = parser.parse_args()

    # Enforce that either both or none are provided
    if (args.fixed_segmentation_path is None) != (args.moving_segmentation_path is None):
        raise ValueError("Arguments --fixed_segmentation_path and --moving_segmentation_path must be given together or not at all.")

    if args.use_mask:
        args.fixed_segmentation_path = Path(args.fixed_path).parent / "mask.nii.gz"
        args.moving_segmentation_path = Path(args.moving_path).parent / "mask.nii.gz"

    # Check if unigradicon-register is installed
    assert shutil.which('unigradicon-register') is not None, "unigradicon-register is not installed."

    # Validate paths
    validate_path(args.fixed_path, "Fixed image path")
    validate_path(args.moving_path, "Moving image path")
    validate_path(args.fixed_segmentation_path, "Fixed segmentation path")
    validate_path(args.moving_segmentation_path, "Moving segmentation path")

    # Get the full composite output suffix
    moving_dir = Path(args.moving_path).parent
    fixed_dir = Path(args.fixed_path).parent
    parts = list(moving_dir.parts)
    try:
        moving_handle = "_".join([parts[parts.index("data") + 1]] + parts[parts.index("processed") + 1:])
    except ValueError as e:
        raise ValueError(f"Expected 'data' and 'processed' directories to be in path: {args.moving_path}") from e
    suffix = f"_{args.suffix}" if args.suffix else ""
    comp_suffix = f"{moving_handle}_model_{args.model}_io_{args.io_iterations}_sim_{args.io_sim}{suffix}"

    # Create output directory
    output_dir = fixed_dir.parent / "unigradicon"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the moving image basename
    moving_path = Path(args.moving_path)
    moving_basename = moving_path.name.removesuffix(''.join(moving_path.suffixes))

    # Set default paths
    defaults = {
        "transform_out": output_dir / f"registered_{moving_basename}_template_{comp_suffix}.hdf5",
        "warped_moving_out": output_dir / f"warped_{moving_basename}_template_{comp_suffix}.nii.gz",
    }
    set_default_paths(args, defaults)

    # Get absolute paths
    for attr in [
        "fixed_path", "fixed_segmentation_path", "moving_path",
        "moving_segmentation_path", "transform_out", "warped_moving_out"
    ]:
        path = getattr(args, attr)
        setattr(args, attr, str(Path(path).resolve()) if path else None)

    # Run the registration
    with Timer():
        run_registration(args)


if __name__ == "__main__":
    main()
