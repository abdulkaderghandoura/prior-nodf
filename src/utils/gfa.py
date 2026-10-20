import os
import sys
import argparse

import torch
import numpy as np
from dipy.reconst.odf import gfa
from dipy.data import get_sphere

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utility import *


def compute_gfa(
    odf_sh_coeffs: torch.Tensor,
    mask: np.ndarray,
    affine: np.ndarray,
    sh_order: int,
    output_path: str,
) -> np.ndarray:
    """Compute GFA from ODFs using DIPY."""
    with Timer():
        # Get ODFs
        sphere = get_sphere("repulsion724")
        B = compute_sh_basis(sphere.vertices, sh_order)[0]  # (P, K)
        odfs_flat = odf_sh_coeffs[mask].cpu().detach().numpy() @ B.T  # (N, P)

        # Compute GFA
        gfa_flat = gfa(odfs_flat)  # (N,)
        gfa_flat = np.nan_to_num(gfa_flat, nan=0.0)
        gfa_image = np.zeros(mask.shape)
        gfa_image[mask] = gfa_flat

    # Save GFA as a NIfTI image
    save_nifti(gfa_image, affine, output_path)
    print(f"✔  GFA saved to {os.path.dirname(output_path)}", "\n")

    return gfa_image


def main():
    parser = argparse.ArgumentParser(description="Compute GFA from ODFs using DIPY.")
    parser.add_argument("--data_dir", required=True, help="Path to the data directory")
    parser.add_argument("--sh_order", type=int, default=8, help="Spherical harmonic order")
    parser.add_argument("--suffix", type=str, help="Suffix to include in filename")
    parser.add_argument("--roi", nargs=6, metavar=('x1', 'x2', 'y1', 'y2', 'z1', 'z2'),
                        type=int, help="Region of interest to compute GFA for")
    args = parser.parse_args()

    # Create paths
    suffix = f"_{args.suffix}" if args.suffix else ""
    mask_path = os.path.join(args.data_dir, "mask.nii.gz")
    gt_dir = os.path.join(args.data_dir, "gt_calcs")
    odf_sh_coeffs_path = os.path.join(gt_dir, f"odf_sh_coeffs{suffix}.pt")
    output_path = os.path.join(gt_dir, f"gfa{suffix}.nii.gz")

    # Load mask and affine
    mask, affine = get_mask(mask_path, roi=args.roi)[:2]

    # Load ODF SH coefficients
    odf_sh_coeffs = get_odf_sh_coeffs(path=odf_sh_coeffs_path, mask=mask)  # (X, Y, Z, K)

    # Compute GFA
    print(f"==> Computing GFA...")
    compute_gfa(
        odf_sh_coeffs=odf_sh_coeffs,
        mask=mask,
        affine=affine,
        sh_order=args.sh_order,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
