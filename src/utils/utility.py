import os
import sys
import time
import random
import itertools
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Union

import torch
import numpy as np
import pytorch_lightning as pl
import nibabel as nib
from scipy.special import legendre, gamma
from dipy.reconst.shm import real_sh_descoteaux
from dipy.core.sphere import Sphere, disperse_charges, HemiSphere
from dipy.segment.mask import applymask
from dipy.align.reslice import reslice
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import get_bval_indices


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

def set_seed(seed: int, reproducible: bool = False):
    """Set a fixed seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # More strict reproducibility (might slow down training!)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_default_paths(
    args: argparse.Namespace,
    defaults: Dict[str, str],
) -> None:
    """Set default paths for the given arguments."""
    for attr, path in defaults.items():
        setattr(args, attr, path)


def get_experiment_version_dir(args: argparse.Namespace) -> str:
    """Get the experiment version directory."""
    assert args.experiment_version is not None, \
        'The experiment_version arg must be provided!'
    experiment_version = "version_" + str(args.experiment_version)
    return os.path.join(
        args.output_dir,
        args.experiment_name,
        args.model,
        experiment_version
    )

def create_grid(
    size: Tuple[int, int, int],
    step: Tuple[int, int, int] = (1, 1, 1),
    mask: torch.Tensor = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
) -> torch.Tensor:
    """Create a grid of points inside the image."""
    continuous = not all(isinstance(s, int) for s in step)

    x = torch.arange(0, size[0] + 1, step[0], device=device)[:mask.shape[0] if mask is not None else None]
    y = torch.arange(0, size[1] + 1, step[1], device=device)[:mask.shape[1] if mask is not None else None]
    z = torch.arange(0, size[2] + 1, step[2], device=device)[:mask.shape[2] if mask is not None else None]
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

    if mask is None:
        mask = torch.ones(xx.shape, device=device).bool()

    assert xx.shape == yy.shape == zz.shape == mask.shape, \
        f"xx.shape: {xx.shape}, yy.shape: {yy.shape}, zz.shape: {zz.shape}, and mask.shape: {mask.shape} must be the same!"

    grid = torch.stack(
        [xx[mask], yy[mask], zz[mask]], dim=-1
    ).reshape(-1, 3).to(torch.float32 if continuous else torch.int32)  # (N, 3)

    if verbose:
        label = "continuous" if continuous else "discrete"
        print(f"Created a {label} grid of {grid.shape[0]} points in the {size} image with step {step}.")

    return grid

def create_voxel_subgrid(
    centers: torch.Tensor,
    size: Tuple[int, int, int],
    corners_only: bool = False,
) -> torch.Tensor:
    """Generate a subgrid of points within each voxel, excluding the boundaries."""

    # Generate the 8 corners of each voxel
    if corners_only:
        offsets = torch.tensor(
            list(itertools.product([-0.5, 0.5], repeat=3)),
            dtype=torch.float32,
            device=centers.device
        )  # (8, 3)

        corners = centers[:, None, :] + offsets[None, :, :]
        return corners  # (N, 8, 3)

    vx, vy, vz = size
    dx = torch.linspace(-0.5 + 0.5 / vx, 0.5 - 0.5 / vx, vx, device=centers.device)
    dy = torch.linspace(-0.5 + 0.5 / vy, 0.5 - 0.5 / vy, vy, device=centers.device)
    dz = torch.linspace(-0.5 + 0.5 / vz, 0.5 - 0.5 / vz, vz, device=centers.device)

    dxx, dyy, dzz = torch.meshgrid(dx, dy, dz, indexing='ij')
    subgrid_offsets = torch.stack([dxx, dyy, dzz], dim=-1).reshape(-1, 3)
    subgrid_points = centers[:, None, :] + subgrid_offsets[None, :, :]


    return subgrid_points  # (N, vx * vy * vz, 3)


def load_nifti(
    file_path: str,
    is_mask: bool = False,
    description: str = "NIfTI file",
    verbose: bool = True,
) -> Tuple[np.array, np.array, Tuple[float, float, float], nib.Nifti1Header]:
    """Load NIfTI file and return data, affine, header, and zooms."""
    try:
        with Timer():
            nii_img = nib.load(file_path)
            data = nii_img.get_fdata()
            affine = nii_img.affine.copy()
            zooms = nii_img.header.get_zooms()[:3]
            header = nii_img.header.copy()
            if is_mask:
                data = data.astype(bool)
            if verbose:
                print(f"Loaded {description} of shape {data.shape} from:\n {file_path}")
    except Exception as e:
        raise ValueError(f"Error loading {description} from path '{file_path}': {e}")

    return data, affine, zooms, header


def reslice_data(
    data: np.array,
    affine: np.array,
    old_zooms: Tuple[float, float, float],
    new_zooms: Tuple[float, float, float],
    order: int = 1,
) -> Tuple[np.array, np.array]:
    """Reslice data using new zooms and return new data and affine."""
    new_data, new_affine = reslice(data, affine, old_zooms, new_zooms, order=order)
    return new_data, new_affine


def save_nifti(
    data: np.array,
    affine: np.array,
    output_path: str,
    description: str = "NIfTI file",
    verbose: bool = True,
) -> None:
    """Save data to a NIfTI file."""
    new_img = nib.Nifti1Image(data, affine)
    new_img.update_header()
    try:
        with Timer():
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            nib.save(new_img, output_path)
            if verbose:
                print(f"Saved {description} of shape {data.shape} to:", "\n", output_path)
    except Exception as e:
        print(f"Error saving {description}: {e}")


def save_torch_tensor(
    data: torch.Tensor,
    output_path: str,
    description: str = "torch tensor",
    verbose: bool = True,
) -> None:
    """Save torch tensor to a file."""
    try:
        with Timer():
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(data, output_path)
            if verbose:
                print(f"Saved {description} of shape {data.shape} to:", "\n", output_path)
    except Exception as e:
        print(f"Error saving {description}: {e}")


def nan_to_num_tensor(
    tensor: torch.Tensor,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0
) -> torch.Tensor:
    """Replace nan, inf, and -inf with the given values."""
    tensor = tensor.clone()
    tensor[torch.isnan(tensor)] = nan
    tensor[tensor == float('inf')] = posinf
    tensor[tensor == float('-inf')] = neginf
    return tensor


def reslice_to_reference_shape(data, ref_spatial_shape):
    """Cropping / padding data to match the reference spatial shape."""
    assert data.ndim >= 3, f"Invalid data shape: {data.shape}"

    # Compute full target shape, preserving the number of diffusion directions if exists
    full_target_shape = tuple(ref_spatial_shape) + data.shape[3:]
    result = np.zeros(full_target_shape, dtype=data.dtype)

    # Compute overlap for spatial axes (slice from index 0 to s along each axis)
    min_spatial_shape = np.minimum(data.shape[:3], ref_spatial_shape)
    spatial_slices = tuple(slice(0, s) for s in min_spatial_shape)

    # Add a tuple of slice(None) that is equivalent to [:] for the angular dim if exists
    full_slices = spatial_slices + (slice(None),) * (data.ndim - 3)

    result[full_slices] = data[full_slices]
    return result


def get_odf_sh_coeffs(
    path: str,
    mask: np.ndarray,
) -> torch.Tensor:
    """Load ODF spherical harmonic coefficients from the given path"""
    # Load pointwise estimates
    try:
        with Timer():
            pointwise_estimate = torch.load(path, map_location="cpu", weights_only=False).float()  # (N, K)
            print(f"Loaded ODF spherical harmonic coefficients of shape: {pointwise_estimate.shape} from:\n {path}")
    except Exception as e:
        print(f"Error loading ODF spherical harmonic coefficients: {e}")

    # Create ODF spherical harmonic coefficients tensor
    odf_sh_coeffs = torch.zeros((*mask.shape, pointwise_estimate.shape[-1]))  # (X, Y, Z, K)
    odf_sh_coeffs[mask] = pointwise_estimate  # (X, Y, Z, K)

    return odf_sh_coeffs


def get_bvals_bvecs(
    bval_path: str,
    bval: float,
    tolerance: float,
    bvec_path: str = None,
    M: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load b-values and b-vectors from the given paths."""
    # Load b-values and b-vectors
    bvals, bvecs = read_bvals_bvecs(fbvals=bval_path, fbvecs=bvec_path)
    print(f"Loaded b-values from:\n {bval_path}\n")
    if bvec_path is not None:
        print(f"Loaded b-vectors from:\n {bvec_path}\n")

    # Select b-value shell indices within the range (bval ± tolerance)
    bval_indices = get_bval_indices(bvals=bvals, bval=bval, tol=tolerance)

    # Select M gradient directions
    M = len(bval_indices) if M is None else M
    print(f"Selected {M} out of {len(bval_indices)} gradient directions.\n")

    # Filter b-values and b-vectors
    bval_indices_filtered = bval_indices[: M]
    bvals_filtered = bvals[bval_indices_filtered]
    bvecs_filtered = bvecs[bval_indices_filtered] if bvec_path else None

    return bvals_filtered, bvecs_filtered, bval_indices_filtered


def normalize_signal(
    signal: np.ndarray,
    mask: np.ndarray,
    data_dir: str,
    bval: float,
    tolerance: float,
    M: int = None,
    fine_tune: bool = False,
) -> np.ndarray:
    """Normalize and clip signal."""
    # Load b-values from the original data (not from a custom bval_path)
    bval_path = os.path.join(data_dir, "bval.txt")
    bval_indices = get_bvals_bvecs(bval_path=bval_path, bval=bval, tolerance=tolerance, M=M)[2]

    # Load b0 mean
    signal_b0_mean_path = os.path.join(data_dir, "b0.nii.gz")
    signal_b0_mean = load_nifti(signal_b0_mean_path)[0]

    # Normalize and clip signal
    with Timer():
        print("==> Normalizing and clipping signal...")
        signal_normed = signal[..., bval_indices] / (signal_b0_mean[..., np.newaxis] + 1e-6)
        signal_normed = np.nan_to_num(signal_normed, nan=1e-6, posinf=1.0, neginf=1e-6)
        signal_normed = applymask(signal_normed, mask)  # (X, Y, Z, M)
        signal_normed = np.clip(signal_normed, 0, 1)

        # Check if the number of b-values matches the number of signal volumes
        assert len(bval_indices) == signal_normed.shape[-1], (
            f"Mismatch between number of b-values {len(bval_indices)} "
            f"and number of normalized signal volumes {signal_normed.shape[-1]}"
        )

        # Get the path to the calculations directory
        calcs_dir = get_calcs_path(data_dir, fine_tune)

        # Save normalized and clipped signal
        signal_normed_path = os.path.join(calcs_dir, f"signal_normed_M_{len(bval_indices)}.npy")
        np.save(signal_normed_path, signal_normed[mask])
        print(f"Saved normalized and clipped signal to:\n{signal_normed_path}\n")

    return signal_normed


def compute_sh_basis(
    vertices: np.ndarray,
    sh_order: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute real symmetric spherical harmonic basis for a set of gradient directions."""
    spherical_coords = cart2sphere(vertices)
    theta = spherical_coords[:, 0]  # azimuthal (longitudinal) coordinate
    phi = spherical_coords[:, 1]    # polar (colatitudinal) coordinate

    sh_basis, m_values, l_values = real_sh_descoteaux(sh_order, theta, phi, legacy=True)
    return sh_basis, m_values, l_values


def matern_spec_density(omega: float, rho: float, nu: float):
    """
    Spectral density for a Matern covariance function. Form can be found in Dutordoir et. al 2020 supplement.

    omega: frequency
    rho: lengthscale
    nu: differentiability
    """
    term1 = (
        (2**3) * (np.pi ** (3 / 2)) * gamma(nu + (3 / 2)) * np.power(2 * nu, nu)
    ) / (gamma(nu) * np.power(rho, 2 * nu))
    term2 = np.power(
        ((2 * nu) / np.power(rho, 2)) + (4 * (np.pi**2) * np.power(omega, 2)),
        -(nu + (3 / 2)),
    )
    return term1 * term2

def get_odf_transformation(n: np.ndarray):
    """
    Get the transformation matrix for the ODF signal.

    n: spherical harmonic order
    """
    T = np.zeros((len(n), len(n)))
    for i in range(T.shape[0]):
        P_n = legendre(n[i])
        T[i, i] = P_n(0)
    return T


def get_signal_transformation(n: np.ndarray):
    """
    Get the transformation matrix for the signal.

    n: spherical harmonic order
    """
    Tinv = np.zeros((len(n), len(n)))
    for i in range(Tinv.shape[0]):
        P_n = legendre(n[i])
        Tinv[i, i] = 1.0 / P_n(0)
    return Tinv


def cart2sphere(x):
    """
    Convert cartesian to spherical coordinates.

    x: cartesian coordinates
    """
    r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)
    theta = np.arccos(x[:, 2] / r)
    phi = np.arctan2(x[:, 1], x[:, 0])
    return np.column_stack([theta, phi])

def get_phi_r_tensors(args):
    """
    Get Phi and R tensors.

    args: arguments
    ODFSPACE: flag to use ODF space

    returns:
        Phi_tensor: torch.Tensor (K x M)
        R_tensor: (M x M)
    """
    print(f"\n==> Getting Phi and R tensors for {args.bval} b-value...")

    # Load b-vectors
    bvecs = get_bvals_bvecs(
        bval_path=args.bval_path,
        bval=args.bval,
        tolerance=args.tolerance,
        bvec_path=args.bvec_path,
        M=args.M,
    )[1]

    # Function mapping: convert b-vectors to spherical coordinates
    sphere = Sphere(xyz=bvecs)

    # Compute the spherical harmonic basis functions
    B, _, n = compute_sh_basis(sphere.vertices, args.sh_order)

    # Define harmonic function space: get transformation matrices
    T_n_inv = get_signal_transformation(n)

    Phi_tensor = torch.from_numpy(B @ T_n_inv).float()

    R_tensor = torch.from_numpy(
        np.diag(1 / matern_spec_density(np.sqrt(n * (n + 1)), args.rho, args.nu))
    ).float()

    return Phi_tensor, R_tensor


def get_file_name_ckpt_suffix(args: argparse.Namespace) -> Tuple[str, str]:
    """Get the checkpoint name and suffix for the file name."""
    # Initialize an empty suffix
    final_suffix = ""
    ckpt_name = None

    # Add the zooms to the suffix if provided
    if args.zooms:
        final_suffix += f"_resliced_{'x'.join(str(z).replace('.', 'p') for z in args.zooms)}"

    # Add the spatial reference image to the suffix if provided (in addition to the zooms suffix!)
    if args.reslice_like:
        p = Path(args.reslice_like)
        parts = list(p.parts if p.is_dir() else p.parent.parts)
        final_suffix += f"_like_{'_'.join([parts[parts.index('data') + 1]] + parts[parts.index('processed') + 1:])}"

    # Add the angular reference image to the suffix if provided (in addition to the zooms suffix!)
    if args.angular_res_like:
        parts = list(Path(args.angular_res_like).parts)
        final_suffix += f"_angular_res_like_{'_'.join([parts[parts.index('data') + 1]] + parts[parts.index('processed') + 1:])}"

    # Add a custom suffix if provided
    if args.suffix:
        final_suffix += f"_{args.suffix}"

    # Get the checkpoint name
    if args.ckpt_path:
        ckpt_name = os.path.splitext(os.path.basename(args.ckpt_path))[0]

    return ckpt_name, final_suffix


def reslice_img(
    input_path: str,
    new_zooms: Tuple[float, float, float] = None,
    reslice_like: str = None,
    order: int = 1
):
    """Reslice the image to either the specified zooms or as the specified image."""
    assert new_zooms is not None or reslice_like is not None, \
        "Either new zooms or reslice like must be provided!"

    # Load NIfTI
    data, affine, zooms = load_nifti(input_path)[:3]

    # Perform reslicing
    data_resliced, affine_resliced = reslice(data, affine, zooms, new_zooms, order=order)
    print(f"Resliced image with zooms {zooms} to zooms {new_zooms}")

    # Match shape to the reference if provided
    if reslice_like:
        ref_spatial_shape = load_nifti(reslice_like)[0].shape[:3]
        print(f"Matching shape {data_resliced.shape} to the reference with shape {ref_spatial_shape}")
        data_resliced = reslice_to_reference_shape(data_resliced, ref_spatial_shape)

    return data_resliced, affine_resliced

def reslice_mask(
    input_path: str,
    new_zooms: Tuple[float, float, float] = None,
    reslice_like: str = None,
    mask_output_path: str = None
):
    """Reslice the mask to either the specified zooms or as the specified image."""
    # Reslice the mask
    data_resliced, affine_resliced = reslice_img(
        input_path,
        new_zooms=new_zooms,
        reslice_like=reslice_like,
        order=1
    )

    # Convert to boolean mask
    data_resliced = (data_resliced > 0.5).astype(np.uint8)

    # Save the resliced mask if an output path is provided
    if mask_output_path:
        save_nifti(data_resliced, affine_resliced, mask_output_path)

    return data_resliced.astype(bool)


def get_mask(
    mask_path: str,
    full_mask: bool = False,
    roi: Tuple[int, int, int, int, int, int] = None,
) -> np.array:
    """
    Loads the mask and applies custom modifications to it if needed.
    """
    mask, affine, zooms, header = load_nifti(mask_path)
    mask = mask.astype(bool)

    if full_mask:
        print(f"Number of voxels in mask: {mask.sum()}\n")
        return mask, affine, zooms, header

    def _inclusive_slice(start, stop):
        if stop == -1:
            return slice(start, None)
        return slice(start, stop + 1)

    if roi:
        x1, x2, y1, y2, z1, z2 = roi
        region_mask = np.zeros_like(mask)
        region_mask[
            _inclusive_slice(x1, x2),
            _inclusive_slice(y1, y2),
            _inclusive_slice(z1, z2)
        ] = 1
        mask = mask * region_mask


    print(f"Number of voxels in mask: {mask.sum()}\n")
    return mask, affine, zooms, header


def get_calcs_path(data_dir: str, fine_tune: bool = False) -> str:
    """Get the path to the calculations directory."""
    dir_name = "finetune_calcs" if fine_tune else "train_calcs"
    return os.path.join(data_dir, dir_name)


def get_args(cmd: bool = True):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description="NODF Estimation."
    )

    parser.add_argument(
        "--output_dir",
        default=os.path.join("output", "experiments"),
        type=str,
        help="Path to directory to store output.",
    )

    parser.add_argument(
        "--log_dir",
        default="logs",
        type=str,
        help="Path to directory to store logs and hyperparameters.",
    )

    parser.add_argument(
        "--data_dir",
        default="data",
        type=str,
        help="Path to data directory containing signal.nii.gz, mask.nii.gz, bval.txt, and bvec.txt.",
    )

    parser.add_argument(
        "--model",
        default="nodf",
        type=str,
        help="Name of the model.",
    )

    parser.add_argument(
        "--ckpt_path",
        default=None,
        type=str,
        help="Path to checkpoint file.",
    )

    parser.add_argument(
        "--odf_sh_coeffs_path",
        default=None,
        type=str,
        help="Path to ODF spherical harmonic coefficients file.",
    )

    parser.add_argument(
        "--ref_data_dir",
        default=None,
        type=str,
        help="Path to the high-resolution data directory.",
    )

    parser.add_argument(
        "--experiment_name",
        default="scratch",
        type=str,
        help="Name of the experiment.",
    )

    parser.add_argument(
        "--inr",
        default="siren",
        type=str,
        choices=["siren", "wire", "relu"],
        help="Type of the INR - wire or siren or relu",
    )

    parser.add_argument(
        "--bval",
        default=1000,
        type=float,
        help="Target b-value shell (in s/mm²) to extract indices.",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
        choices=["cpu", "cuda"],
        help='Device to use: "cpu" or "cuda"',
    )

    parser.add_argument(
        "--sh_order",
        default=8,
        type=int,
        help="Order of spherical harmonic basis",
    )

    parser.add_argument(
        "--tolerance",
        default=20,
        type=int,
        help="+= tolerance considered same b-value.",
    )

    parser.add_argument(
        "--rho",
        default=0.5,
        type=float,
        help="Length-scale parameter for Matern Prior.",
    )

    parser.add_argument(
        "--nu",
        default=1.5,
        type=float,
        help="Smoothness parameter for Matern Prior.",
    )

    parser.add_argument(
        "--num_epochs",
        default=10000,
        type=int,
        help="Number of trainging epochs.",
    )

    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="Learning rate for optimizer.",
    )

    parser.add_argument(
        "--r",
        default=128,
        type=int,
        help="Rank of spatial basis.",
    )

    parser.add_argument(
        "--depth",
        default=3,
        type=int,
        help="Number of hidden layers.",
    )


    parser.add_argument(
        "--batch_size",
        default=65536,
        type=int,
        help="Batch size.",
    )


    parser.add_argument(
        "--lambda_c",
        default=3.76e-7,
        type=float,
        help="Prior regularization strength.",
    )

    parser.add_argument(
        "--transform",
        type=str,
        help="Path to the HDF5 file",
    )

    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="Set the mode to fine-tuning.",
    )

    parser.add_argument(
        '--moving',
        type=str,
        help='Path to the moving baseline image',
    )

    parser.add_argument(
        '--fixed',
        type=str,
        help='Path to the fixed baseline image',
    )

    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Force preprocessing.",
    )

    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Set True to use continuous index coordinates.",
    )

    parser.add_argument(
        "--downsampling_subgrid_size",
        nargs=3,
        type=int,
        default=None,
        help='Subgrid size for one voxel to be used for downsampling.',
    )

    parser.add_argument(
        "--num_downsampling_points",
        default=6,
        type=int,
        help="Number of random points to use for downsampling.",
    )

    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Set True to pin memory.",
    )

    # One of two debug options can be provided if needed
    debug_opts = parser.add_mutually_exclusive_group()

    debug_opts.add_argument(
        "--debug",
        action="store_true",
        help="Disable callbacks and directories initialization.",
    )

    debug_opts.add_argument(
        "--debug_with_tensorboard",
        action="store_true",
        help="Disable callbacks and directories initialization, but use TensorBoard logger.",
    )

    parser.add_argument(
        "--profiler",
        action="store_true",
        help="Set True to use profiler.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set True to print verbose output.",
    )


    parser.add_argument(
        "--warmup_epochs",
        default=40,
        type=int,
        help="Number of warmup epochs.",
    )

    parser.add_argument(
        "--omega0",
        default=30.0,
        type=float,
        help="WIRE Gaussian parameter.",
    )

    parser.add_argument(
        "--omega0_hidden",
        default=30.0,
        type=float,
        help="WIRE Gaussian parameter for hidden layers.",
    )

    parser.add_argument(
        "--sigma0",
        default=5.0,
        type=float,
        help="WIRE Sine frequency parameter.",
    )

    parser.add_argument(
        "--skip_conn",
        action="store_true",
        help="Set True to use skip connections.",
    )

    parser.add_argument(
        "--batchnorm",
        action="store_true",
        help="Set True to use batch normalization.",
    )

    parser.add_argument(
        "--num_workers",
        default=12,
        type=int,
        help="Number of workers for the dataloader.",
    )

    parser.add_argument(
        "--n_levels",
        default=4,
        type=int,
        help="Number of resolution levels for hash encoding.",
    )

    parser.add_argument(
        "--n_features_per_level",
        default=8,
        type=int,
        help="Number of features per embeddings vector.",
    )

    parser.add_argument(
        "--log2_hashmap_size",
        default=20,
        type=int,
        help="Hash map size 2**log2_hashmap_size.",
    )

    parser.add_argument(
        "--base_resolution",
        default=80,
        type=int,
        help="Base resolution for hash encoding.",
    )

    parser.add_argument(
        "--per_level_scale",
        default=1.13,
        type=float,
        help="Per level scal of resolution.",
    )

    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay.",
    )

    parser.add_argument(
        "--roi",
        nargs=6,
        metavar=('x1', 'x2', 'y1', 'y2', 'z1', 'z2'),
        type=int,
        help="Region of interest (inclusive ranges)"
    )

    parser.add_argument(
        "--M",
        default=None,
        type=int,
        help="Number of gradient directions.",
    )

    parser.add_argument(
        "--use_baseline",
        action="store_true",
        help="Set True to use the baseline NODF model.",
    )

    parser.add_argument(
        "--tv_directions",
        default=None,
        type=str,
        choices=["x", "y", "z", "xy", "yz", "xz", "xyz"],
        help="Apply total variation regularization in the specified spatial direction(s).",
    )

    parser.add_argument(
        "--lambda_tv",
        default=0.4,
        type=float,
        help="Total variation prior strength.",
    )

    parser.add_argument(
        "--offset_tv",
        default=0.001,
        type=float,
        help="Offset for approximating total variation.",
    )

    parser.add_argument(
        "--gradient_clip_val",
        default=0.0,
        type=float,
        help="Gradient clipping value.",
    )

    parser.add_argument(
        "--snapshot_epochs",
        nargs="+",
        default=[2, 4, 9, 20, 30, 40, 50, 75, 99, 150, 200, 300, 400],
        type=int,
        help="List of additional checkpoints to save.",
    )

    parser.add_argument(
        "--snapshot_minutes",
        nargs="+",
        default=[2, 5, 10, 30, 60, 120, 180],
        type=int,
        help="List of additional time-based checkpoints to save (in minutes).",
    )

    parser.add_argument(
        "--every_n_epochs",
        default=500,
        type=int,
        help="Save checkpoint every n epochs.",
    )

    parser.add_argument(
        "--zooms",
        nargs=3,
        default=None,  # NEVER GIVE IT A DEFAULT VALUE
        type=float,
        metavar=("X", "Y", "Z"),
        help="Zooms for x, y, z dimensions, space-separated"
    )

    parser.add_argument(
        "--reslice_like",
        default=None,  # NEVER GIVE IT A DEFAULT VALUE
        type=str,
        help="Path to a reference image for zooms"
    )

    parser.add_argument(
        "--angular_res_like",
        default=None,
        type=str,
        help="Path to a reference directory containing b-values and b-vectors for angular resolution.",
    )

    parser.add_argument(
        '--freeze_mlp_except_last',
        default=None,
        type=int,
        help='Freeze all but the last specified number of layers.',
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        '--suffix',
        default=None,
        type=str,
        help='Custom suffix for the file name.',
    )

    parser.add_argument(
        '--weights_only',
        action='store_true',
    )

    parser.add_argument(
        '--disable_weights_only',
        action='store_false',
        dest='weights_only',
    )

    parser.set_defaults(weights_only=True)

    parser.add_argument(
        '--reproducible',
        action='store_true',
        help='Set True only when reproducibility is needed (it slows down training).',
    )


    args = parser.parse_args()

    if args.reslice_like:
        assert os.path.exists(args.reslice_like), \
            f"--reslice_like path {args.reslice_like} does not exist!"

    if args.fine_tune and args.experiment_name == "scratch":
        raise ValueError("Provide a value for --experiment_name other than the " \
                          "default 'scratch' when using --fine_tune.")

    if args.ref_data_dir is None:
        args.ref_data_dir = os.path.dirname(args.reslice_like) if args.reslice_like else args.data_dir

    # Set default file names
    defaults = {
        "signal_path": os.path.join(args.data_dir, "signal.nii.gz"),
        "mask_path": os.path.join(args.data_dir, "mask.nii.gz"),
        "bval_path": os.path.join(args.data_dir, "bval.txt"),
        "bvec_path": os.path.join(args.data_dir, "bvec.txt"),
        "b0_path": os.path.join(args.data_dir, "b0.nii.gz"),
    }

    # Set default ground truth file names
    gt_defaults = {
        "gt_odf_sh_coeffs_path": os.path.join(args.ref_data_dir, "gt_calcs", "odf_sh_coeffs.pt"),
        "gt_gfa_path": os.path.join(args.ref_data_dir, "gt_calcs", "gfa.nii.gz"),
        "gt_dti_path": os.path.join(args.ref_data_dir, "gt_calcs", "dti.nii.gz"),
        "gt_fodfs_path": os.path.join(args.ref_data_dir, "gt_calcs", "fodfs_tournier07.nii.gz"),
        "gt_md_path": os.path.join(args.ref_data_dir, "gt_calcs", "md.nii.gz"),
        "gt_fa_path": os.path.join(args.ref_data_dir, "gt_calcs", "fa.nii.gz"),
        "gt_evecs_path": os.path.join(args.ref_data_dir, "gt_calcs", "evecs.nii.gz"),
        "gt_tractogram_path": os.path.join(args.ref_data_dir, "gt_calcs", "tractography_csd_bv3_FB_ode.tck"),
    }

    # Set default paths
    set_default_paths(args, defaults)
    set_default_paths(args, gt_defaults)

    # Use the baseline image as moving for training from scratch
    if not args.moving:
        args.moving = args.b0_path

    # Assign the zooms from the reference image if provided
    if args.reslice_like:
        args.zooms = load_nifti(args.reslice_like)[2]
        print(f"Using zooms {args.zooms} from {args.reslice_like}")

    return args


def update_args(args):
    # get explicitly passed arguments
    explicit_args = {arg: getattr(args, arg) for arg in vars(args) if f"--{arg}" in sys.argv}

    # set args to model's args and update them
    args = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)["hyper_parameters"]["args"]
    vars(args).update(explicit_args)

    if not 'preprocess' in explicit_args:
        args.preprocess = False

    # Update all args that depend on the data path (e.g., in case of the 0-iteration checkpoint)
    if "data_dir" in explicit_args:

        # Set default file names
        defaults = {
            "signal_path": os.path.join(args.data_dir, "signal.nii.gz"),
            "mask_path": os.path.join(args.data_dir, "mask.nii.gz"),
            "bval_path": os.path.join(args.data_dir, "bval.txt"),
            "bvec_path": os.path.join(args.data_dir, "bvec.txt"),
            "b0_path": os.path.join(args.data_dir, "b0.nii.gz"),
        }
        set_default_paths(args, defaults)

    if args.ref_data_dir is None:
        args.ref_data_dir = os.path.dirname(args.reslice_like) if args.reslice_like else args.data_dir

    # Set default ground truth file names
    gt_defaults = {
        "gt_odf_sh_coeffs_path": os.path.join(args.ref_data_dir, "gt_calcs", "odf_sh_coeffs.pt"),
        "gt_gfa_path": os.path.join(args.ref_data_dir, "gt_calcs", "gfa.nii.gz"),
        "gt_dti_path": os.path.join(args.ref_data_dir, "gt_calcs", "dti.nii.gz"),
        "gt_fodfs_path": os.path.join(args.ref_data_dir, "gt_calcs", "fodfs_tournier07.nii.gz"),
        "gt_md_path": os.path.join(args.ref_data_dir, "gt_calcs", "md.nii.gz"),
        "gt_fa_path": os.path.join(args.ref_data_dir, "gt_calcs", "fa.nii.gz"),
        "gt_evecs_path": os.path.join(args.ref_data_dir, "gt_calcs", "evecs.nii.gz"),
        "gt_tractogram_path": os.path.join(args.ref_data_dir, "gt_calcs", "tractography_csd_bv3_FB_ode.tck"),
    }
    set_default_paths(args, gt_defaults)

    if args.angular_res_like:
        assert os.path.isdir(args.angular_res_like), \
            f"--angular_res_like path {args.angular_res_like} must be a directory!"

        # Get the bval and bvec paths
        args.bval_path = os.path.join(args.angular_res_like, "bval.txt")
        args.bvec_path = os.path.join(args.angular_res_like, "bvec.txt")

        # Undo reading M from a checkpoint when we have new b-values and b-vectors
        if 'M' not in explicit_args:
            args.M = None

    # Assign the zooms from the reference image if provided
    if "reslice_like" in explicit_args:
        args.zooms = load_nifti(args.reslice_like)[2]
        print(f"Using zooms {args.zooms} from {args.reslice_like}")

    return args
