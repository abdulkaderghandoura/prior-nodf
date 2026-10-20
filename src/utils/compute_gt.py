import os
import sys
import argparse
import threading
from typing import Union, List, Callable
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.shls import compute_shls
from utils.gfa import compute_gfa
from utils.dti import compute_dti
from utils.fodfs import compute_fodfs
from utils.tractography import compute_tractography
from utils.utility import set_default_paths, set_seed, get_odf_sh_coeffs, get_mask


def keepalive(paths: Union[Path, List[Path]], interval: int) -> Callable[[], None]:
    """Keep remote filesystem mounts active during long idle periods in processing."""
    if isinstance(paths, Path):
        paths = [paths]

    # Stop signal for worker
    stop = threading.Event()

    # Define the worker thread
    def worker():
        while not stop.is_set():
            for p in paths:
                try:
                    with os.scandir(p) as it:
                        # Read a single entry; cheap
                        next(it, None)
                except Exception:
                    # Swallow transient hiccups
                    pass  # try again next cycle
            stop.wait(interval)

    # Start the worker thread
    threading.Thread(target=worker, daemon=True).start()
    return stop.set  # Call this to stop the keep-alive


def compute_gt(args: argparse.Namespace) -> None:
    """Compute the ground-truth SH coefficients, GFA, DTI metrics, fODFs and tractogram."""
    # Set a fixed seed for reproducibility
    set_seed(args.seed)

    # Keep remote filesystem mounts active during long idle periods in processing
    interval = args.keepalive_minutes * 60
    stop_keepalive = keepalive([args.output_dir], interval=interval)

    # Set suffix and ground truth paths
    suffix = f"_{args.suffix}" if args.suffix else ""
    odf_sh_coeffs_path = os.path.join(args.output_dir, f"odf_sh_coeffs{suffix}.pt")
    gfa_path = os.path.join(args.output_dir, f"gfa{suffix}.nii.gz")
    fa_img_path = os.path.join(args.output_dir, f"fa{suffix}.nii.gz")
    fodfs_path = os.path.join(args.output_dir, f"fodfs_tournier07{suffix}.nii.gz")

    if "shls" not in args.skip:
        # Compute spherical harmonic coefficients
        print("==> Computing spherical harmonic coefficients...")
        compute_shls(
            data_dir=args.data_dir,
            signal_path=args.signal_path,
            mask_path=args.mask_path,
            bval_path=args.bval_path,
            bvec_path=args.bvec_path,
            bval=args.bval,
            tolerance=args.tolerance,
            M=args.M,
            sh_order=args.sh_order,
            output_dir=args.output_dir,
            suffix=suffix,
            roi=args.roi,
        )

    if "gfa" not in args.skip:
        # Load mask and affine
        mask, affine = get_mask(args.mask_path, roi=args.roi)[:2]

        # Load ODF spherical harmonic coefficients
        odf_sh_coeffs = get_odf_sh_coeffs(odf_sh_coeffs_path, mask)  # (X, Y, Z, K)

        # Compute GFA
        print("==> Computing GFA...")
        compute_gfa(
            odf_sh_coeffs=odf_sh_coeffs,
            mask=mask,
            affine=affine,
            sh_order=args.sh_order,
            output_path=gfa_path,
        )

    if "dti" not in args.skip:
        # Compute DTI metrics
        print("==> Computing DTI metrics...")
        compute_dti(
            data_dir=args.data_dir,
            signal_path=args.signal_path,
            mask_path=args.mask_path,
            bval_path=args.bval_path,
            bvec_path=args.bvec_path,
            bval=args.bval,
            tolerance=args.tolerance,
            M=args.M,
            suffix=suffix,
            output_dir=args.output_dir,
            erosion_iter=args.erosion_iter,
            roi=args.roi,
        )

    if "fodfs" not in args.skip:
        # Compute fODFs
        print("==> Computing fODFs...")
        compute_fodfs(
            odf_sh_coeffs_path=odf_sh_coeffs_path,
            mask_path=args.mask_path,
            sh_order=args.sh_order,
            output_path=fodfs_path,
            roi=args.roi,
        )

    if "tractography" not in args.skip:
        # Compute tractography
        print("==> Computing tractography...")
        compute_tractography(
            fodfs_path=fodfs_path,
            mask_path=args.mask_path,
            fa_img_path=fa_img_path,
            suffix=suffix,
            output_dir=args.output_dir,
            roi=args.roi,
        )

    # Stop the keepalive thread
    stop_keepalive()


def main():
    parser = argparse.ArgumentParser(
        description="Compute the ground-truth SH coefficients, GFA, DTI metrics, fODFs and tractogram"
    )

    # Required arguments
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to the data directory"
    )
    parser.add_argument(
        "--bval",
        type=float,
        required=True,
        help="Target b-value"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        required=True,
        help="Margin around the target b-value"
    )

    # Optional arguments
    parser.add_argument(
        "--signal_path",
        help="Path to the signal NIfTI file"
    )
    parser.add_argument(
        "--mask_path",
        help="Path to the mask NIfTI file"
    )
    parser.add_argument(
        "--bval_path",
        help="Path to the b-values file"
    )
    parser.add_argument(
        "--bvec_path",
        help="Path to the b-vectors file"
    )
    parser.add_argument(
        "--M",
        type=int,
        help="Number of gradient directions to use"
    )
    parser.add_argument(
        "--sh_order",
        default=8,
        type=int,
        help="Order of spherical harmonic basis"
    )
    parser.add_argument(
        "--erosion_iter",
        type=int,
        help="Number of iterations for binary erosion"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--suffix",
        help="Suffix to include in filename"
    )
    parser.add_argument(
        "--output_dir",
        help="Path to the output directory"
    )

    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=["shls", "gfa", "dti", "fodfs", "tractography"],
        help="Skip the following ground truth computations"
    )

    # Use MRtrix (or a similar tool) to determine the region of interest visually
    parser.add_argument(
        "--roi",
        nargs=6,
        metavar=('x1', 'x2', 'y1', 'y2', 'z1', 'z2'),
        type=int,
        help="Region of interest to compute ground truth for (inclusive ranges)"
    )

    parser.add_argument(
        "--keepalive_minutes",
        type=int,
        default=5,
        help="Keepalive interval in minutes"
    )

    # Parse CLI arguments
    args = parser.parse_args()

    # Set default paths (if not provided)
    defaults = {
        "signal_path": os.path.join(args.data_dir, "signal.nii.gz"),
        "mask_path": os.path.join(args.data_dir, "mask.nii.gz"),
        "bval_path": os.path.join(args.data_dir, "bval.txt"),
        "bvec_path": os.path.join(args.data_dir, "bvec.txt"),
        "output_dir": os.path.join(args.data_dir, "gt_calcs"),
    }
    set_default_paths(args, defaults)

    # Compute ground truth
    compute_gt(args)


if __name__ == "__main__":
    main()
