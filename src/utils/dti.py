import os
import sys
import argparse
from typing import Tuple

import numpy as np
from dipy.reconst import dti
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import fractional_anisotropy, color_fa
from scipy.ndimage import binary_erosion

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utility import *


def compute_dti(
    data_dir: str,
    signal_path: str,
    mask_path: str,
    bval_path: str,
    bvec_path: str,
    bval: float,
    tolerance: float,
    M: int = None,
    suffix: str = None,
    output_dir: str = None,
    ckpt_name: str = None,
    erosion_iter: int = None,
    fine_tune: bool = False,
    normed: bool = False,
    roi: Tuple[int, int, int, int, int, int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Diffusion Tensor Imaging (DTI) metrics from diffusion MRI data."""
    # Load signal and mask
    signal, signal_affine = load_nifti(signal_path)[:2]
    mask = get_mask(mask_path, roi=roi)[0]

    assert signal.shape[:3] == mask.shape
    print(f"Number of voxels in mask: {mask.sum()}")

    # Load b-values and b-vectors
    bvals, bvecs, _ = get_bvals_bvecs(
        bval_path=bval_path,
        bvec_path=bvec_path,
        M=M,
        bval=bval,
        tolerance=tolerance,
    )

    # Normalize signal by b0
    if not normed:
        signal = normalize_signal(
            signal=signal,
            mask=mask,
            data_dir=data_dir,
            bval=bval,
            tolerance=tolerance,
        )

    # Add b0 volume
    print("==> Adding b0 volume of ones...")
    with Timer():
        signal_b0_mean = np.ones((*mask.shape, 1), dtype=np.float32)  # b0/b0
        signal = signal * signal_b0_mean  # the signal is b0-normalised, so b0 = 1
        signal_expanded = np.concatenate([signal_b0_mean, signal], axis=-1)  # (X, Y, Z, M + 1)

        # Prepend zero b-value and b-vector for b0
        bvals = np.insert(bvals, 0, 0)
        bvecs = np.insert(bvecs, 0, [0, 0, 0], axis=0)

    assert signal_expanded.shape[3] == bvals.shape[0] == bvecs.shape[0], \
        f"Number of signal volumes ({signal_expanded.shape[3]}), " \
        f"b-values ({bvals.shape[0]}) and " \
        f"b-vectors ({bvecs.shape[0]}) must match"

    # Create gradient table
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)

    # Fit the tensor model
    print("==> Fitting tensor model with weighted least squares...")
    with Timer():
        tenmodel = dti.TensorModel(gtab, fit_method='WLS')
        tenfit = tenmodel.fit(signal_expanded, mask=mask)

    # Compute anisotropy measures
    print("==> Computing anisotropy measures (FA, MD, EVECS, RGB)...")
    with Timer():
        FA = np.nan_to_num(fractional_anisotropy(tenfit.evals), nan=0)  # set nan to 0
        tensor_fa = FA.astype(np.float32)
        tensor_evecs = tenfit.evecs.astype(np.float32)
        tensor_md = tenfit.md.astype(np.float32)
        rgb = color_fa(np.clip(FA, 0, 1), tenfit.evecs)

        if erosion_iter:
            assert erosion_iter > 0, "Erosion iterations must be greater than 0"
            mask = binary_erosion(mask, iterations=erosion_iter)
            rgb[~mask] = 0  # clean up bright edges

        tensor_rgb = (255 * rgb).astype(np.uint8)

    # Save DTI metrics
    os.makedirs(output_dir, exist_ok=True)
    fname_ext = f"{'_' + ckpt_name if ckpt_name else ''}{suffix}.nii.gz"
    save_nifti(tensor_fa, signal_affine, os.path.join(output_dir, f"fa{fname_ext}"))
    save_nifti(tensor_evecs, signal_affine, os.path.join(output_dir, f"evecs{fname_ext}"))
    save_nifti(tensor_md, signal_affine, os.path.join(output_dir, f"md{fname_ext}"))
    save_nifti(tensor_rgb, signal_affine, os.path.join(output_dir, f"dti{fname_ext}"))

    print(f"✔  DTI metrics saved to:\n {output_dir}\n")

    return tensor_fa, tensor_evecs, tensor_md, tensor_rgb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute DTI metrics from diffusion MRI data using DIPY."
    )

    # Required arguments
    parser.add_argument(
        "--data_dir",
        type=str,
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
        type=str,
        help="Path to the signal NIfTI file"
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        help="Path to the mask NIfTI file"
    )
    parser.add_argument(
        "--bval_path",
        type=str,
        help="Path to the b-values file"
    )
    parser.add_argument(
        "--bvec_path",
        type=str,
        help="Path to the b-vectors file"
    )
    parser.add_argument(
        "--M",
        type=int,
        help="Number of gradient directions"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        help="Suffix for the output file name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory"
    )
    parser.add_argument(
        "--erosion_iter",
        type=int,
        help="Number of iterations for binary erosion"
    )

    # Use MRtrix (or a similar tool) to determine the region of interest visually
    parser.add_argument(
        "--roi",
        nargs=6,
        metavar=('x1', 'x2', 'y1', 'y2', 'z1', 'z2'),
        type=int,
        help="Region of interest to compute DTI for"
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

    # Compute DTI metrics
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


if __name__ == "__main__":
    main()
