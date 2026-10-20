import os
import shutil
import sys
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import nibabel as nib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utility import Timer


def set_default_paths(args: argparse.Namespace, defaults: Dict[str, str]) -> None:
    """Set default paths for the given arguments."""
    for key, default_name in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, Path(args.data_dir) / default_name)


def block_mean(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Average slices along the Z axis in blocks.
    Drop trailing slices that don't fill a block.
    """
    nx, ny, nz = data.shape[:3]
    n_blocks = nz // factor
    if n_blocks < 1:
        raise ValueError("Z is smaller than factor")

    n_slices = n_blocks * factor
    if n_slices != nz:
        # Build a list of ":" for all dimensions
        idx = [slice(None)] * data.ndim
        # Trim to [0:n_slices] along Z
        idx[2] = slice(0, n_slices)
        # Apply slices to array
        data = data[tuple(idx)]

    # Reshape to group Z into (n_blocks, factor), then mean over the factor axis
    data = np.ascontiguousarray(data)  # ensure C-order before reshape
    new_shape = (*data.shape[:2], n_blocks, factor, *data.shape[3:])
    data_reshaped = data.reshape(new_shape)
    return data_reshaped.mean(axis=3)


def scale_affine(affine: np.ndarray, factor: int) -> np.ndarray:
    """
    Increase voxel size along Z by the given factor and
    shift origin so slice centers match the average of each block.
    """
    new_affine = affine.copy()
    # Shift origin so slice centers match the mean of each block
    new_affine[:3, 3] += ((factor - 1) / 2.0) * affine[:3, 2]
    # Scale z-direction vector by factor
    new_affine[:3, 2] *= factor
    return new_affine


def save_like(
    ref_img: nib.Nifti1Image,
    data: np.ndarray,
    affine: np.ndarray,
    path: str,
    is_mask: bool = False
) -> None:
    """Save NIfTI with ref header dtype and updated sform/qform."""
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.uint8 if is_mask else np.float32)
    data = data.astype(np.uint8 if is_mask else np.float32, copy=False)

    # Neutralize scaling to avoid double-scaling on read
    hdr['scl_slope'] = 1
    hdr['scl_inter'] = 0

    out = nib.Nifti1Image(data, affine, hdr)

    # Preserve sform/qform codes exactly
    _, sc = ref_img.get_sform(coded=True)
    _, qc = ref_img.get_qform(coded=True)
    out.set_sform(affine, int(sc))
    out.set_qform(affine, int(qc))

    # Set zooms from affine; preserve time zoom if present, else default to 1.0
    vx = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))
    ref_zooms = ref_img.header.get_zooms()
    if len(out.shape) == 4:
        tz = float(ref_zooms[3]) if len(ref_zooms) >= 4 else 1.0
        out.header.set_zooms((*vx, tz))
    else:
        out.header.set_zooms(vx)

    nib.save(out, str(path))


def downsample_throughplane(args: argparse.Namespace) -> None:
    """Downsample through-plane data."""
    with Timer():
        # Load data
        print(f"==> Loading data...")
        signal_nii = nib.load(args.signal)
        mask_nii = nib.load(args.mask)
        b0_nii = nib.load(args.b0)

    if len(signal_nii.shape) != 4:
        raise ValueError("signal (DWI) must be 4D (X,Y,Z,T)")
    if len(b0_nii.shape) != 3:
        raise ValueError("b0 must be 3D (X,Y,Z)")
    if len(mask_nii.shape) != 3:
        raise ValueError("mask must be 3D (X,Y,Z)")

    if mask_nii.shape[:3] != signal_nii.shape[:3] or \
        b0_nii.shape[:3] != signal_nii.shape[:3]:
        raise ValueError("b0, mask, and signal must have the same shape")

    # Scale affine
    aff_signal = scale_affine(signal_nii.affine, args.factor)
    aff_b0 = scale_affine(b0_nii.affine, args.factor)
    aff_mask = scale_affine(mask_nii.affine, args.factor)

    if not np.allclose(signal_nii.affine, b0_nii.affine) or \
        not np.allclose(signal_nii.affine, mask_nii.affine):
        print("Warning: Input affines differ; scaling each image's affine separately.")

    with Timer():
        print(f"==> Downsampling data...")
        # Downsample b0
        b0_ds = block_mean(np.asanyarray(b0_nii.dataobj), factor=args.factor)

        # Downsample mask (binarize -> block mean of {0,1} -> majority threshold)
        mask = np.asanyarray(mask_nii.dataobj)
        mask = (mask >= 0.5).astype(np.uint8)
        mask_ds = block_mean(mask, factor=args.factor)
        mask_ds = (mask_ds >= 0.5).astype(np.uint8)

        # Downsample signal
        signal_ds = block_mean(np.asanyarray(signal_nii.dataobj), factor=args.factor)

    # Save data
    with Timer():
        print(f"==> Saving data...")
        output_dir = Path(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        save_like(signal_nii, signal_ds, aff_signal, output_dir / 'signal.nii.gz')
        save_like(mask_nii, mask_ds, aff_mask, output_dir / 'mask.nii.gz')
        save_like(b0_nii, b0_ds, aff_b0, output_dir / 'b0.nii.gz')

        # Copy bvals and bvecs
        shutil.copy(args.bvals, output_dir / 'bval.txt')
        shutil.copy(args.bvecs, output_dir / 'bvec.txt')

    # Create output subdirectories
    (output_dir / "train_calcs").mkdir(parents=True, exist_ok=True)
    (output_dir / "finetune_calcs").mkdir(parents=True, exist_ok=True)

    print(f"✔  Downsampled data saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Slice-averaging spatial downsampling of DWI, b0, and mask"
    )
    # Required arguments
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Path to data directory'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Path to output directory'
    )

    # Optional arguments
    parser.add_argument('--signal', help='Path to DWI NIfTI file')
    parser.add_argument('--b0', help='Path to b0 NIfTI file')
    parser.add_argument('--mask', help='Path to mask NIfTI file')
    parser.add_argument('--bvals', help='Path to bvals file')
    parser.add_argument('--bvecs', help='Path to bvecs file')
    parser.add_argument(
        '--factor',
        default=4,
        type=int,
        choices=[2, 4],
        help='Downsampling factor (default: 4)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Default file names
    defaults = {
        'signal': 'signal.nii.gz',
        'b0': 'b0.nii.gz',
        'mask': 'mask.nii.gz',
        'bvals': 'bval.txt',
        'bvecs': 'bvec.txt',
    }

    # Set default paths
    set_default_paths(args, defaults)

    # Ensure that all files exist
    for file in [args.signal, args.b0, args.mask, args.bvals, args.bvecs]:
        if file is not None and not Path(file).is_file():
            raise FileNotFoundError(f"File {file} not found in {args.data_dir}")

    downsample_throughplane(args)


if __name__ == '__main__':
    main()
