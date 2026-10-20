import os
import sys
import argparse
from typing import Tuple

import numpy as np
from dipy.reconst.csdeconv import odf_sh_to_sharp
from dipy.reconst.shm import sf_to_sh
from dipy.data import get_sphere

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utility import *


def compute_fodfs(
    odf_sh_coeffs_path: str,
    mask_path: str,
    sh_order: int,
    output_path: str,
    roi: Tuple[int, int, int, int, int, int] = None,
) -> np.ndarray:
    """Sharpen ODFs into fODFs by spherical deconvolution and save them in the Tournier07 basis."""
    # Load mask and affine
    mask0, affine = get_mask(mask_path)[:2]

    # Load ODF spherical harmonic coefficients
    odf_sh_coeffs = get_odf_sh_coeffs(odf_sh_coeffs_path, mask0).numpy()  # (X, Y, Z, K)

    mask = get_mask(mask_path, roi=roi)[0]
    assert mask.shape == mask0.shape and mask.dtype==bool
    odf_sh_coeffs_flat = odf_sh_coeffs[mask]  # (N, K)

    # Get spherical harmonics basis function evaluation matrix
    sphere = get_sphere("repulsion724")

    # Sharpen ODF coefficients
    print("==> Doing Spherical Deconvolution...")
    with Timer():
        fodf_sh_coeffs_flat = odf_sh_to_sharp(
            odf_sh_coeffs_flat,
            sphere,
            basis="descoteaux07",
            ratio=0.3,
            sh_order=sh_order,
            lambda_=1.0,
            tau=0.1,
            r2_term=False,
        )

        # Get ODF evaluations on the sphere
        B = compute_sh_basis(sphere.vertices, sh_order)[0]  # (P, K)
        fodfs_flat = fodf_sh_coeffs_flat @ B.T  # (N, P)

    # Convert to the Tournier07 SH basis used by MRtrix
    fodfs_flat_tourn = sf_to_sh(
        fodfs_flat,
        sphere,
        sh_order=sh_order,
        basis_type="tournier07",
    )

    # Normalize coefficients
    norm = np.linalg.norm(fodfs_flat_tourn, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0  # avoid division by zero
    fodfs_flat_tourn = fodfs_flat_tourn / norm

    # Save fODFs in Tournier07 basis
    print("==> Saving fODFs in Tournier07 basis ...")
    fodfs_tourn = np.zeros((*mask.shape, fodfs_flat_tourn.shape[-1]))
    fodfs_tourn[mask] = fodfs_flat_tourn
    fodfs_tourn = fodfs_tourn.astype(np.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_nifti(fodfs_tourn, affine, output_path)

    print("✔  Saved fODFs in Tournier07 basis successfully!", "\n")

    return fodfs_tourn


def main():
    parser = argparse.ArgumentParser(description="fODF Processing Script")
    parser.add_argument("--data_dir", required=True, help="Path to data directory")
    parser.add_argument("--odf_sh_coeffs_path", help="Path to odf_sh_coeffs.pt")
    parser.add_argument("--mask_path", help="Path to mask NIfTI file")
    parser.add_argument("--sh_order", type=int, default=8, help="Spherical harmonics order")
    parser.add_argument("--suffix", help="Suffix to include in filename")
    parser.add_argument("--output_dir", help="Directory to save output")

    # Use MRtrix (or a similar tool) to determine the region of interest visually
    parser.add_argument("--roi", nargs=6, metavar=('x1', 'x2', 'y1', 'y2', 'z1', 'z2'),
                        type=int, help="Region of interest to compute fODFs for")

    args = parser.parse_args()

    # Set suffix
    suffix = f"_{args.suffix}" if args.suffix else ""

    # Set default paths
    defaults = {
        "mask_path": os.path.join(args.data_dir, "mask.nii.gz"),
        "output_dir": os.path.join(args.data_dir, "gt_calcs"),
        "odf_sh_coeffs_path": os.path.join(args.data_dir, "gt_calcs", f"odf_sh_coeffs{suffix}.pt"),
    }
    set_default_paths(args, defaults)

    # Set output path
    output_path = os.path.join(args.output_dir, f"fodfs_tournier07{suffix}.nii.gz")

    # Compute fODFs
    compute_fodfs(
        odf_sh_coeffs_path=args.odf_sh_coeffs_path,
        mask_path=args.mask_path,
        sh_order=args.sh_order,
        output_path=output_path,
        roi=args.roi,
    )


if __name__ == "__main__":
    main()
