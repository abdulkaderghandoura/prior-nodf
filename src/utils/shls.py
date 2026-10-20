import os
import sys
import argparse
from typing import Tuple

import torch
import numpy as np
from dipy.core.sphere import Sphere
from dipy.reconst.shm import sf_to_sh

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utility import *


def compute_shls(
    data_dir: str,
    signal_path: str,
    mask_path: str,
    bval_path: str,
    bvec_path: str,
    bval: float,
    tolerance: float,
    M: int,
    sh_order: int,
    output_dir: str,
    suffix: str,
    roi: Tuple[int, int, int, int, int, int] = None,
) -> None:
    """Compute spherical harmonic coefficients."""
    # Load signal and mask
    print("==> Loading data...")
    signal = load_nifti(signal_path)[0]
    mask = get_mask(mask_path, roi=roi)[0]

    # Get b-values and b-vectors
    bvals, bvecs, _ = get_bvals_bvecs(
        bval_path=bval_path,
        bvec_path=bvec_path,
        M=M,
        bval=bval,
        tolerance=tolerance,
    )

    # Normalize and clip signal
    signal_normed = normalize_signal(
        signal=signal,
        mask=mask,
        data_dir=data_dir,
        bval=bval,
        tolerance=tolerance,
    )

    # Compute transformation matrix
    sphere = Sphere(xyz=bvecs)
    n = compute_sh_basis(sphere.vertices, sh_order)[2]
    T_n = get_odf_transformation(n)

    # Compute spherical harmonic coefficients
    print("==> Converting signal samples to SH coefficients...")
    with Timer():
        signal_sh_coeffs = sf_to_sh(
            sf=signal_normed[mask],
            sphere=sphere,
            sh_order=sh_order,
            basis_type="descoteaux07",
            smooth=1e-3,
        )

        # Apply the Funk–Radon transform analytically in SH space
        odf_sh_coeffs = signal_sh_coeffs @ T_n

    # Save spherical harmonic coefficients
    output_path = os.path.join(output_dir, f"odf_sh_coeffs{suffix}.pt")
    odf_sh_coeffs = torch.from_numpy(odf_sh_coeffs.astype(np.float32))
    save_torch_tensor(odf_sh_coeffs, output_path, description="ODF SH coeffs")

    print(f"✔  Completed SHLS computation successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Compute SHLS from diffusion MRI data using DIPY."
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

    # Use MRtrix (or a similar tool) to determine the region of interest visually
    parser.add_argument(
        "--roi",
        nargs=6,
        metavar=('x1', 'x2', 'y1', 'y2', 'z1', 'z2'),
        type=int,
        help="Region of interest to compute SHLS for"
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

    # Set suffix
    suffix = f"_{args.suffix}" if args.suffix else ""

    # Set a fixed seed for reproducibility
    set_seed(args.seed)

    # Compute spherical harmonic coefficients
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


if __name__ == "__main__":
    main()
