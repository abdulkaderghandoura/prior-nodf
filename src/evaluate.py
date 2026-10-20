import os
import threading
from typing import Tuple, Dict, Callable, Union, List
from pathlib import Path

import torch
import numpy as np
import json
from skimage.metrics import peak_signal_noise_ratio as _psnr, structural_similarity as _ssim
from image_similarity_measures.quality_metrics import fsim
from dipy.data import default_sphere
from dipy.reconst.shm import sh_to_sf
from dipy.direction import peak_directions

from utils.dti import compute_dti
from utils.gfa import compute_gfa
from utils.fodfs import compute_fodfs
from utils.tractography import compute_tractography
from utils import *


class Evaluation:
    """Evaluates diffusion MRI super-resolution performance."""
    def __init__(self, args):
        self.args = args
        self.calcs_path = get_calcs_path(args.data_dir, args.fine_tune)
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.ckpt_name, self.suffix = get_file_name_ckpt_suffix(args)
        print("\n==> Loading data...")

        # Get experiment version directory
        experiment_version_dir = get_experiment_version_dir(self.args)

        # Load mask
        self.mask_path = self.args.mask_path
        if self.args.zooms:
            self.mask_path = os.path.join(self.calcs_path, f"mask{self.suffix}.nii.gz")
        self.mask_full = get_mask(self.mask_path, full_mask=True)[0]  # X x Y x Z
        self.mask_full = (self.mask_full > 1e-6)

        # Get ground truth mask
        assert self.args.ref_data_dir is not None, \
            "High-resolution data directory (--ref_data_dir) must be specified for evaluation"

        print(f"Using ground truth data from:\n {self.args.ref_data_dir}\n")
        self.gt_mask_path = os.path.join(self.args.ref_data_dir, "mask.nii.gz")
        if self.gt_mask_path == self.mask_path:
            self.gt_mask_full = self.mask_full
        else:
            self.gt_mask_full = get_mask(self.gt_mask_path, full_mask=True)[0]

        self.gt_mask_full = (self.gt_mask_full > 1e-6)

        # Get paths
        self.reconstructed_signal_path = os.path.join(
            experiment_version_dir, "reconstruction",
            f'recon_signal_without_b0_{self.ckpt_name}{self.suffix}.nii.gz'
        )

        self.odf_sh_coeffs_path = os.path.join(
            experiment_version_dir, "prediction",
            f'odf_sh_coeffs_{self.ckpt_name}{self.suffix}.pt'
        ) if not self.args.odf_sh_coeffs_path else self.args.odf_sh_coeffs_path

        self.fodfs_path = os.path.join(
            experiment_version_dir, "evaluation",
            f"fodfs_tournier07_{self.ckpt_name}{self.suffix}.nii.gz"
        )

        self.gfa_path = os.path.join(
            experiment_version_dir, "evaluation",
            f"gfa_{self.ckpt_name}{self.suffix}.nii.gz"
        )

        self.fa_path = os.path.join(
            experiment_version_dir, "evaluation",
            f"fa_{self.ckpt_name}{self.suffix}.nii.gz"
        )
        self.evecs_path = os.path.join(
            experiment_version_dir, "evaluation",
            f"evecs_{self.ckpt_name}{self.suffix}.nii.gz"
        )

        self.tractogram_path = os.path.join(
            experiment_version_dir, "evaluation",
            f"tractography_csd_bv3_FB_ode_{self.ckpt_name}{self.suffix}.tck"
        )

        self.cv_metrics_json_path = os.path.join(
            experiment_version_dir, "evaluation",
            f"cv_metrics_{self.ckpt_name}{self.suffix}.json"
        )

        # Load affine from the reconstructed signal
        self.affine = load_nifti(self.reconstructed_signal_path)[1]
        print(f"Loaded affine from {self.reconstructed_signal_path}\n")

        # Load ODF SH coefficients
        self.odf_sh_coeffs = get_odf_sh_coeffs(
            path=self.odf_sh_coeffs_path,
            mask=self.mask_full,
        ).to(self.device)  # (X, Y, Z, K)

        # Load ground truth ODF spherical harmonic coefficients
        self.gt_odf_sh_coeffs = get_odf_sh_coeffs(
            path=self.args.gt_odf_sh_coeffs_path,
            mask=self.gt_mask_full,
        ).to(self.device) if self.args.ref_data_dir else None

        # Load ground truth GFA and FA
        self.gt_gfa = load_nifti(self.args.gt_gfa_path)[0]
        self.gt_fa = load_nifti(self.args.gt_fa_path)[0]
        self.gt_fodf = load_nifti(self.args.gt_fodfs_path)[0]

        # Compute separate white matter masks for GFA-based and FA-based maps
        self.gt_wm_mask_gfa = self._compute_wm_mask(
            map=self.gt_gfa,
            map_name="GFA",
            thresh=self.args.gfa_wm_thresh
        )

        self.gt_wm_mask_fa = self._compute_wm_mask(
            map=self.gt_fa,
            map_name="FA",
            thresh=self.args.fa_wm_thresh
        )

        # Save white matter masks for inspection
        self.output_dir = os.path.join(experiment_version_dir, "evaluation")
        os.makedirs(self.output_dir, exist_ok=True)
        self._save_wm_masks()

        print(f"Output directory: {self.output_dir}\n")

    def _compute_wm_mask(
        self,
        map: np.ndarray,
        map_name: str,
        thresh: float = 0.20
    ) -> np.ndarray:
        """
        Compute WM mask from anisotropy map (FA or GFA).

        Normalizes map to [0, 1] before applying threshold to ensure
        consistency with dMRI literature standard (threshold ≥ 0.2 for WM).

        Args:
            map: Anisotropy volume (X, Y, Z)
            map_name: Name for logging ("FA" or "GFA")
            thresh: Threshold after normalization (default: 0.20)

        Returns:
            wm_mask: Boolean array (X, Y, Z)
        """
        # Get map values within brain
        map_in_brain = map[self.gt_mask_full]
        map_finite = map_in_brain[np.isfinite(map_in_brain)]

        if len(map_finite) == 0:
            print(f"WARNING: No valid {map_name} values found!")
            return np.zeros_like(self.gt_mask_full, dtype=bool)

        # Normalize to [0, 1]
        m_min = float(map_finite.min())
        m_max = float(map_finite.max())

        print(f"\n{map_name}-based WM mask computation:")
        print(f"  Original {map_name} range: [{m_min:.4f}, {m_max:.4f}]")

        if m_max - m_min < 1e-6:
            print(f"WARNING: {map_name} has no variation!")
            return np.zeros_like(self.gt_mask_full, dtype=bool)

        # Normalize to [0, 1]
        map_norm = (map - m_min) / (m_max - m_min)

        # Apply threshold
        wm_mask = (self.gt_mask_full &
                   np.isfinite(map_norm) &
                   (map_norm >= thresh))

        n_vox = int(wm_mask.sum())
        n_brain = int(self.gt_mask_full.sum())
        thresh_orig = thresh * (m_max - m_min) + m_min

        print(f"  Normalized to: [0.0000, 1.0000]")
        print(f"  WM threshold: {thresh:.2f} (normalized) = {thresh_orig:.4f} (original)")
        print(f"  WM voxels: {n_vox} ({100*n_vox/n_brain:.1f}% of brain)")

        if n_vox > 0:
            wm_map_norm = map_norm[wm_mask]
            wm_map_orig = map[wm_mask]
            print(f"  WM {map_name} (original): [{wm_map_orig.min():.4f}, {wm_map_orig.max():.4f}]")
            print(f"  WM {map_name} (normalized): [{wm_map_norm.min():.4f}, {wm_map_norm.max():.4f}]")
            print(f"  WM {map_name} mean ± std: {wm_map_norm.mean():.4f} ± {wm_map_norm.std():.4f} (normalized)")

        return wm_mask

    def _save_wm_masks(self):
        """Save WM masks for inspection."""
        save_nifti(
            data=self.gt_wm_mask_fa.astype(np.uint8),
            affine=self.affine,
            output_path=os.path.join(
                self.output_dir,
                f"gt_wm_mask_fa{self.args.fa_wm_thresh:.2f}_{self.ckpt_name}{self.suffix}.nii.gz"
            ),
            description=f"GT WM Mask (FA ≥ {self.args.fa_wm_thresh})"
        )

        save_nifti(
            data=self.gt_wm_mask_gfa.astype(np.uint8),
            affine=self.affine,
            output_path=os.path.join(
                self.output_dir,
                f"gt_wm_mask_gfa{self.args.gfa_wm_thresh:.2f}_{self.ckpt_name}{self.suffix}.nii.gz"
            ),
            description=f"GT WM Mask (GFA ≥ {self.args.gfa_wm_thresh})"
        )

    def get_odf_error(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate ODF error using L2-norm.
        Measures how well the spherical harmonic coefficients match ground truth.

        Uses GFA-based WM mask (ODF-derived metric → GFA-based mask).

        Returns:
            errors_median: torch.Tensor (scalar) - median L2 error
            errors: torch.Tensor (X x Y x Z) - per-voxel errors
        """
        # Use GFA-based mask (ODF metrics → GFA mask)
        mask = self.gt_wm_mask_gfa

        # Get ODF SH coefficients as flat tensors
        odf_sh_coeffs_flat = self.odf_sh_coeffs[mask].cpu().detach().numpy()  # (N, K)
        gt_odf_sh_coeffs_flat = self.gt_odf_sh_coeffs[mask].cpu().detach().numpy()  # (N, K)

        # Calculate normalized L2-norm error
        odf_sh_coeffs_diff = odf_sh_coeffs_flat - gt_odf_sh_coeffs_flat
        gt_norm = np.linalg.norm(gt_odf_sh_coeffs_flat, ord=2, axis=-1)
        gt_norm[gt_norm == 0] = 1e-6  # Avoid division by zero

        errors = np.linalg.norm(odf_sh_coeffs_diff, ord=2, axis=-1) / gt_norm

        errors = torch.from_numpy(errors).float().to(self.device)
        errors_median = torch.median(errors)

        print(f"\nODF L2-Norm Error (GFA-based WM mask):")
        print(f"  Median Error: {errors_median:.6f}")
        print(f"  N voxels: {len(errors)}")

        # Create error map
        odf_sh_coeffs_l2_norm_error_img = torch.zeros(mask.shape).to(self.device)
        odf_sh_coeffs_l2_norm_error_img[mask] = errors

        # Save results
        save_torch_tensor(
            data=errors.cpu().detach(),
            output_path=os.path.join(
                self.output_dir,
                f"odf_sh_coeffs_l2_norm_error_values_{self.ckpt_name}{self.suffix}.pt"
            ),
            description="ODF SH Coeffs L2-Norm Error Values",
        )

        save_nifti(
            data=odf_sh_coeffs_l2_norm_error_img.cpu().detach().numpy(),
            affine=self.affine,
            output_path=os.path.join(
                self.output_dir,
                f"odf_sh_coeffs_l2_norm_error_map_{self.ckpt_name}{self.suffix}.nii.gz"
            ),
            description="ODF SH Coeffs L2-Norm Error Map",
        )

        return errors_median, errors

    def get_gfa_and_abs_error(self) -> float:
        """
        Calculate GFA (Generalized Fractional Anisotropy) error.

        GFA is a scalar measure of ODF anisotropy analogous to FA for DTI.
        Lower error indicates better preservation of tissue anisotropy.

        Uses GFA-based WM mask (evaluating GFA → GFA-based mask).

        Returns:
            abs_errors_median: float - median absolute GFA error
        """
        # Compute GFA for predicted ODFs
        print(f"\n==> Computing GFA for predicted ODFs...")
        self.gfa = compute_gfa(
            self.odf_sh_coeffs, self.mask_full, self.affine,
            self.args.sh_order, self.gfa_path
        )

        assert self.gt_mask_full.shape == self.mask_full.shape, \
            "Ground truth mask and mask must have the same shape"

        # Use GFA-based mask (evaluating GFA → GFA mask)
        gfa_abs_diff = np.abs(self.gfa[self.gt_wm_mask_gfa] - self.gt_gfa[self.gt_wm_mask_gfa])

        abs_errors_median = float(np.median(gfa_abs_diff))
        abs_errors_mean = float(np.mean(gfa_abs_diff))

        print(f"\nGFA Error (GFA-based WM mask):")
        print(f"  Median Absolute Error: {abs_errors_median:.6f}")
        print(f"  Mean Absolute Error: {abs_errors_mean:.6f}")
        print(f"  N voxels: {len(gfa_abs_diff)}")

        return abs_errors_median

    def get_fsim(self):
        """
        Get FSIM score for FA and GFA scalar maps.

        FSIM (Feature Similarity Index) is appropriate for scalar maps.
        Higher is better (range [0,1]).

        Computed on full 2D slices (whole brain) for standard comparison.

        Returns:
            (fa_fsim, fa_vals), (gfa_fsim, gfa_vals)
        """
        print("\n==> Computing FSIM (full slices)...")

        # Load predicted scalar maps as torch tensors
        fa = torch.from_numpy(load_nifti(self.fa_path)[0])
        gfa = torch.from_numpy(load_nifti(self.gfa_path)[0])

        # Load ground truth
        gt_fa = torch.from_numpy(self.gt_fa)
        gt_gfa = torch.from_numpy(self.gt_gfa)

        print("Computing FSIM for FA...")
        fa_fsim, fa_fsim_values = self._get_fsim_score(fa, gt_fa)
        print(f"FSIM FA: {fa_fsim:.4f}")

        print("Computing FSIM for GFA...")
        gfa_fsim, gfa_fsim_values = self._get_fsim_score(gfa, gt_gfa)
        print(f"FSIM GFA: {gfa_fsim:.4f}")

        # Save FSIM values
        torch.save(fa_fsim_values.cpu().detach(),
                os.path.join(self.output_dir, f"fa_fsim_values.pt"))
        torch.save(gfa_fsim_values.cpu().detach(),
                os.path.join(self.output_dir, f"gfa_fsim_values.pt"))

        return (fa_fsim, fa_fsim_values), (gfa_fsim, gfa_fsim_values)

    def _get_fsim_score(
        self,
        pred_img: torch.Tensor,
        gt_img: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        """
        Get the FSIM score for the given images

        pred_img: torch.Tensor (X x Y x Z) or (X x Y x Z x 3)
        gt_img: torch.Tensor (X x Y x Z) or (X x Y x Z x 3)

        returns: float, torch.Tensor (F)
        """
        def _min_max_normalize(img):
            """Helper function to normalize the image to [0, 1]"""
            img = img.float()
            denominator = (img.max() - img.min())
            if denominator == 0:
                denominator = torch.tensor(1.0, device=img.device, dtype=img.dtype)
            return (img - img.min()) / denominator

        # Handle gray scale images
        if len(gt_img.shape) == 3:
            gt_img = (_min_max_normalize(gt_img) * 255).to(torch.uint8).unsqueeze(-1)  # X x Y x Z x 1
            pred_img = (_min_max_normalize(pred_img) * 255).to(torch.uint8).unsqueeze(-1)  # X x Y x Z x 1

        def _compute_fsim_slices(gt, pred, axis):
            """Helper function to compute FSIM over slices along a given axis"""
            scores = []
            for i in range(gt.size(axis)):
                gt_slice = gt.select(axis, i).cpu().numpy()
                pred_slice = pred.select(axis, i).cpu().numpy()
                scores.append(fsim(gt_slice, pred_slice))
            return scores

        # get fsim score for all axial, sagittal, and coronal slices
        values = torch.tensor(
            _compute_fsim_slices(gt_img, pred_img, 0) +
            _compute_fsim_slices(gt_img, pred_img, 1) +
            _compute_fsim_slices(gt_img, pred_img, 2),
            dtype=torch.float32
        )

        # Remove nans coming from black images
        values = values[~torch.isnan(values)]

        # Median over all slices
        median = torch.median(values).item()

        return median, values

    def _minmax01(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize image to [0, 1] range."""
        x = x.float()
        denominator = (x.max() - x.min())
        denominator = denominator if denominator > 0 else torch.tensor(1.0, device=x.device)
        return (x - x.min()) / denominator

    def _iter_slices(self, vol: torch.Tensor):
        """Iterate over all axial, sagittal, and coronal slices."""
        for ax in (0, 1, 2):
            for i in range(vol.size(ax)):
                yield vol.select(ax, i).cpu().numpy()

    def get_psnr(
        self
    ) -> Tuple[Tuple[float, torch.Tensor], Tuple[float, torch.Tensor]]:
        """
        Calculate PSNR (Peak Signal-to-Noise Ratio) in dB.

        Applied to: FA and GFA scalar maps.
        Higher is better. Typical range: 20-40 dB for medical images.

        Computed on full 2D slices (whole brain) for standard comparison.

        Returns:
            (fa_psnr, fa_vals), (gfa_psnr, gfa_vals)
        """
        print("\n==> Computing PSNR (full slices)...")
        # Load predicted scalar maps
        gfa = torch.from_numpy(load_nifti(
            os.path.join(self.output_dir, f"gfa_{self.ckpt_name}{self.suffix}.nii.gz")
        )[0])
        fa = torch.from_numpy(load_nifti(
            os.path.join(self.output_dir, f"fa_{self.ckpt_name}{self.suffix}.nii.gz")
        )[0])

        # Load ground truth
        gt_gfa = torch.from_numpy(self.gt_gfa)
        gt_fa = torch.from_numpy(self.gt_fa)

        out = {}
        for name, pred, gt in [("GFA", gfa, gt_gfa), ("FA", fa, gt_fa)]:
            pred01 = self._minmax01(pred)
            gt01 = self._minmax01(gt)

            vals = []
            for s_gt, s_pr in zip(self._iter_slices(gt01), self._iter_slices(pred01)):
                with np.errstate(divide="ignore", invalid="ignore"):
                    v = _psnr(s_gt, s_pr, data_range=1.0)
                vals.append(v)

            vals = torch.tensor(vals, dtype=torch.float32)
            vals = torch.nan_to_num(vals, posinf=100.0)
            median = vals.median().item() if vals.numel() else float("nan")

            torch.save(vals, os.path.join(self.output_dir, f"{name.lower()}_psnr_values.pt"))
            print(f"PSNR {name}: {median:.2f} dB")
            out[name.lower()] = (median, vals)

        return out["fa"], out["gfa"]

    def get_ssim(
        self,
        data_range: float = 1.0
    ) -> Tuple[Tuple[float, torch.Tensor], Tuple[float, torch.Tensor]]:
        """
        Calculate SSIM (Structural Similarity Index).

        Applied to: FA and GFA scalar maps.
        Range: [0, 1], higher is better. More perceptually relevant than PSNR.

        Computed on full 2D slices (whole brain) for standard comparison.

        Returns:
            (fa_ssim, fa_vals), (gfa_ssim, gfa_vals)
        """
        print("\n==> Computing SSIM (full slices)...")
        gfa = torch.from_numpy(load_nifti(
            os.path.join(self.output_dir, f"gfa_{self.ckpt_name}{self.suffix}.nii.gz")
        )[0])
        fa = torch.from_numpy(load_nifti(
            os.path.join(self.output_dir, f"fa_{self.ckpt_name}{self.suffix}.nii.gz")
        )[0])

        gt_gfa = torch.from_numpy(self.gt_gfa)
        gt_fa = torch.from_numpy(self.gt_fa)

        out = {}
        for name, pred, gt in [("GFA", gfa, gt_gfa), ("FA", fa, gt_fa)]:
            pred01 = self._minmax01(pred)
            gt01 = self._minmax01(gt)

            vals = []
            for s_gt, s_pr in zip(self._iter_slices(gt01), self._iter_slices(pred01)):
                try:
                    # Get the channel axis
                    ch_axis = -1 if s_gt.ndim == 3 else None

                    # Compute SSIM
                    v = _ssim(
                        s_gt, s_pr, data_range=data_range,
                        channel_axis=ch_axis,
                        gaussian_weights=True, sigma=1.5,
                        use_sample_covariance=False
                    )
                except TypeError:
                    # fallback for old skimage
                    v = _ssim(
                        s_gt, s_pr, data_range=data_range,
                        multichannel=(s_gt.ndim == 3),
                        gaussian_weights=True, sigma=1.5,
                        use_sample_covariance=False
                    )
                vals.append(v)

            vals = torch.tensor(vals, dtype=torch.float32)
            vals = torch.nan_to_num(vals)
            median = vals.median().item() if vals.numel() else float("nan")

            torch.save(vals, os.path.join(self.output_dir, f"{name.lower()}_ssim_values.pt"))
            print(f"SSIM {name}: {median:.4f}")
            out[name.lower()] = (median, vals)

        return out["fa"], out["gfa"]

    def get_nrmse(self):
        """
        Calculate NRMSE (Normalized Root Mean Square Error) for FA and GFA.

        NRMSE = RMSE / (max(observed) - min(observed))

        More sensitive to large errors than MAE and scale-invariant across modalities.
        Lower is better. Range: [0, +inf), but typically [0, 1] for good results.

        Returns:
            (fa_nrmse, gfa_nrmse)
        """
        print("\n==> Computing NRMSE (whole brain)...")

        # Load predicted scalar maps
        fa = torch.from_numpy(load_nifti(
            os.path.join(self.output_dir, f"fa_{self.ckpt_name}{self.suffix}.nii.gz")
        )[0])
        gfa = torch.from_numpy(load_nifti(
            os.path.join(self.output_dir, f"gfa_{self.ckpt_name}{self.suffix}.nii.gz")
        )[0])

        # Load ground truth
        gt_fa = torch.from_numpy(self.gt_fa)
        gt_gfa = torch.from_numpy(self.gt_gfa)

        out = {}
        for name, pred, gt in [("FA", fa, gt_fa), ("GFA", gfa, gt_gfa)]:
            brain_mask = torch.tensor(self.gt_mask_full, dtype=torch.bool)
            mask = brain_mask & ((gt > 1e-6) | (pred > 1e-6))

            if mask.sum() == 0:
                print(f"WARNING: No valid voxels for {name}")
                out[name.lower()] = float('nan')
                continue

            pred_masked = pred[mask]
            gt_masked = gt[mask]

            # Calculate RMSE
            mse = torch.mean((pred_masked - gt_masked) ** 2)
            rmse = torch.sqrt(mse)

            # Normalize by range of ground truth
            gt_range = gt_masked.max() - gt_masked.min()

            if gt_range < 1e-6:
                print(f"WARNING: Ground truth range too small for {name}")
                nrmse = float('nan')
            else:
                nrmse = (rmse / gt_range).item()

            print(f"NRMSE {name} (whole brain): {nrmse:.6f}")
            print(f"  RMSE: {rmse.item():.6f}, GT range: [{gt_masked.min():.4f}, {gt_masked.max():.4f}]")
            print(f"  N voxels: {mask.sum().item()}")

            out[name.lower()] = nrmse

        return out["fa"], out["gfa"]

    def get_eigenvector_angular_error(self):
        """
        Compute acute-angle error (0-90°) between principal eigenvectors (v1) of DTI.

        Uses pre-computed eigenvector volumes with shape (X, Y, Z, 3, 3) where
        columns represent eigenvectors ordered by eigenvalue magnitude.
        Principal eigenvector v1 is the first column (largest eigenvalue).

        The acute angle is computed using absolute dot product, which accounts for
        the sign ambiguity in eigenvector direction (v and -v are equivalent).

        Uses FA-based WM mask (eigenvectors from DTI → FA-based mask).

        Returns:
            results_dict: Dict with keys:
                - 'median': float - median angular error in degrees
                - 'mean': float - mean angular error
                - 'std': float - standard deviation
                - 'pct_above_30': float - percentage of voxels >30° error
        """
        print("\n==> Computing Eigenvector Angular Error (FA-based WM mask)...")

        # Load eigenvectors
        pred_evecs, pred_affine = load_nifti(self.evecs_path)[:2]
        gt_evecs, gt_affine = load_nifti(self.args.gt_evecs_path)[:2]

        # Sanity checks
        if pred_evecs.shape != gt_evecs.shape:
            raise ValueError(
                f"Pred and GT evec shapes differ: {pred_evecs.shape} vs {gt_evecs.shape}. "
                "Ensure both are computed in the same space."
            )

        # Expected shape: (X, Y, Z, 3, 3)
        if pred_evecs.ndim != 5 or pred_evecs.shape[-2:] != (3, 3):
            raise ValueError(
                f"Expected eigenvector shape (X, Y, Z, 3, 3), got {pred_evecs.shape}"
            )

        # Use FA-based WM mask (eigenvectors from DTI → FA mask)
        wm_mask = self.gt_wm_mask_fa & np.isfinite(self.gt_fa)

        n_vox = int(wm_mask.sum())
        print(f"  N voxels: {n_vox}")

        if n_vox == 0:
            print("WARNING: No valid white matter voxels for eigenvector comparison.")
            return None

        # Extract principal eigenvectors (first column: largest eigenvalue)
        pred_v1 = pred_evecs[wm_mask, :, 0].reshape(-1, 3)  # (N, 3)
        gt_v1 = gt_evecs[wm_mask, :, 0].reshape(-1, 3)      # (N, 3)

        # Normalize rows (eigenvectors should already be normalized, but ensure it)
        def _norm_rows(v):
            norm = np.linalg.norm(v, axis=1, keepdims=True)
            norm = np.where(norm < 1e-8, 1.0, norm)
            return v / norm

        pred_v1 = _norm_rows(pred_v1)
        gt_v1 = _norm_rows(gt_v1)

        # Compute acute-angle error (0-90°) using absolute dot product
        cosine = np.abs(np.sum(pred_v1 * gt_v1, axis=1))
        cosine = np.clip(cosine, 0.0, 1.0)
        ang_deg = np.degrees(np.arccos(cosine)).astype(np.float32)

        # Compute statistics
        median_error = float(np.median(ang_deg))
        mean_error = float(np.mean(ang_deg))
        std_error = float(np.std(ang_deg))
        pct_above_30 = float((ang_deg > 30).mean() * 100.0)

        print(f"  Median: {median_error:.2f}°")
        print(f"  Mean ± Std: {mean_error:.2f} ± {std_error:.2f}°")
        print(f"  Voxels with error >30°: {pct_above_30:.2f}% (of {n_vox} WM voxels)")

        # Build full map with NaN outside ROI
        ang_map = np.full(wm_mask.shape, np.nan, dtype=np.float32)
        ang_map[wm_mask] = ang_deg

        # Save error values
        np.save(
            os.path.join(
                self.output_dir,
                f"eigenvector_angular_errors_{self.ckpt_name}{self.suffix}.npy"
            ),
            ang_deg
        )

        # Save error map as NIfTI
        save_nifti(
            data=ang_map,
            affine=gt_affine,
            output_path=os.path.join(
                self.output_dir,
                f"eigenvector_angular_error_map_{self.ckpt_name}{self.suffix}.nii.gz"
            ),
            description="Principal Eigenvector Angular Error Map (FA-based WM)"
        )

        # Return results
        results_dict = {
            'median': median_error,
            'mean': mean_error,
            'std': std_error,
            'pct_above_30': pct_above_30
        }

        return results_dict

    def get_fodf_peak_count_error(
        self,
        rel_peak_thresh: float = 0.6,
        abs_peak_thresh: float = 0.15,
        min_sep_angle: float = 35.0
    ):
        """
        Peak-count accuracy and angular error from fODF SH volumes.
        Uses sh_to_sf + peak_directions with relative threshold and min separation.
        Saves count arrays and NIfTI maps.

        Uses GFA-based WM mask (peaks from fODF → GFA-based mask).

        Parameters
        ----------
        rel_peak_thresh : float, default=0.6
            Relative peak threshold (fraction of max peak amplitude).
            Standard range: 0.5-0.7. Higher = more conservative.

        abs_peak_thresh : float, default=0.15
            Absolute peak amplitude threshold (normalized SF units).
            Standard range: 0.1-0.2. Filters low-amplitude noise.
            Applied to L2-normalized fODFs (consistent with Tournier07 basis).

        min_sep_angle : float, default=35.0
            Minimum separation angle (degrees) between peaks.
            Standard range: 25-40°. Higher = merges closer peaks.

        Returns
        -------
        result : dict
            Dictionary containing:
            - n_voxels: Number of WM voxels evaluated
            - mae: Mean absolute error in peak count
            - mae_std: Standard deviation of absolute errors
            - accuracy: Percentage of voxels with exact peak count match
            - over_estimated_pct: Percentage with too many peaks
            - under_estimated_pct: Percentage with too few peaks
            - angular_error_mean_deg: Mean angular error when counts match
            - angular_error_std_deg: Std of angular errors when counts match
            - pred_distribution: Predicted peak count distribution
            - gt_distribution: Ground truth peak count distribution
            - peak_specific_stats: Dict with per-category statistics
            - rel_peak_thresh, abs_peak_thresh, min_separation_angle_deg: Parameters used
        """
        from scipy.optimize import linear_sum_assignment

        print("\n==> Computing fODF Peak Count Error (GFA-based WM mask)...")

        pred_fodf = load_nifti(self.fodfs_path)[0]
        if pred_fodf.shape != self.gt_fodf.shape:
            raise ValueError(f"Predicted and GT fODF shapes differ: "
                            f"{pred_fodf.shape} vs {self.gt_fodf.shape}")

        # Use GFA-based WM mask (peaks from fODF → GFA mask)
        wm_gt = self.gt_wm_mask_gfa & np.isfinite(self.gt_gfa)

        # Create intersection mask - only voxels valid in both GT and pred
        # Load the predicted mask
        pred_mask = get_mask(self.mask_path, full_mask=True)[0] > 1e-6

        # Use intersection: voxels that are WM in GT AND present in predicted mask
        wm = wm_gt & pred_mask

        n_vox = int(wm.sum())

        # Report mask statistics
        n_gt_only = int(wm_gt.sum() - wm.sum())
        print(f"  Mask statistics:")
        print(f"    GT WM mask: {int(wm_gt.sum()):,} voxels")
        print(f"    Pred mask: {int(pred_mask.sum()):,} voxels")
        print(f"    Intersection (used): {n_vox:,} voxels")
        print(f"    GT-only voxels (excluded): {n_gt_only:,} ({100*n_gt_only/wm_gt.sum():.2f}%)")

        if n_vox == 0:
            print("WARNING: No valid voxels for peak count analysis")
            return None

        sphere = default_sphere
        sh_order = int(self.args.sh_order)

        def to_hemisphere(dirs: np.ndarray) -> np.ndarray:
            """Map all directions to upper hemisphere (z >= 0)."""
            if dirs.size == 0:
                return dirs
            d = dirs.copy()
            d[d[:, 2] < 0] *= -1.0
            return d

        def sh_to_sf_masked(sh_vol):
            """Convert SH coefficients to SF values on sphere, only for WM voxels."""
            coeffs = sh_vol[wm]
            return sh_to_sf(coeffs, sphere, sh_order=sh_order, basis_type="tournier07")

        def detect(sf_vals):
            """
            Detect peaks in SF values.

            Returns
            -------
            counts : ndarray (N,)
                Number of peaks per voxel
            dirs_list : list of ndarray
                Peak directions per voxel (N_peaks, 3)
            """
            N = sf_vals.shape[0]
            counts = np.zeros(N, np.int16)
            dirs_list = [None] * N

            for i in range(N):
                sf = sf_vals[i]

                # Check for invalid values
                if not np.all(np.isfinite(sf)):
                    dirs_list[i] = np.empty((0, 3), np.float32)
                    continue

                # Detect peaks using DIPY
                dirs, _, _ = peak_directions(
                    sf.astype(np.float64),
                    sphere,
                    relative_peak_threshold=rel_peak_thresh,
                    min_separation_angle=min_sep_angle,
                )

                # Map to hemisphere and store
                d = to_hemisphere(dirs.astype(np.float32)) if dirs.size else np.empty((0, 3), np.float32)
                counts[i] = d.shape[0]
                dirs_list[i] = d

            return counts, dirs_list

        def mean_ang_err_optimal(gt_dirs, pr_dirs):
            """
            Compute mean angular error using optimal (Hungarian) matching.

            Parameters
            ----------
            gt_dirs : ndarray (N, 3)
                Ground truth peak directions
            pr_dirs : ndarray (N, 3)
                Predicted peak directions

            Returns
            -------
            error : float
                Mean angular error in degrees, or NaN if counts don't match
            """
            if gt_dirs.shape[0] == 0 or gt_dirs.shape[0] != pr_dirs.shape[0]:
                return np.nan

            # Compute cost matrix (angular distance)
            # Use absolute dot product to handle sign ambiguity
            cos_sim = np.abs(gt_dirs @ pr_dirs.T)
            cos_sim = np.clip(cos_sim, 0.0, 1.0)  # Numerical stability
            cost = np.arccos(cos_sim)

            # Optimal assignment (Hungarian algorithm)
            row_ind, col_ind = linear_sum_assignment(cost)

            # Return mean angular error in degrees
            return float(np.mean(np.degrees(cost[row_ind, col_ind])))

        # Detect peaks in both predicted and ground truth
        print("  Detecting peaks...")
        sf_pred = sh_to_sf_masked(pred_fodf)
        sf_gt = sh_to_sf_masked(self.gt_fodf)

        pred_counts, pred_dirs = detect(sf_pred)
        gt_counts, gt_dirs = detect(sf_gt)

        # Calculate peak count metrics
        diff = pred_counts - gt_counts
        abs_diff = np.abs(diff)

        mae = float(np.mean(abs_diff))
        mae_std = float(np.std(abs_diff))

        acc = float(np.mean(diff == 0) * 100.0)
        over = float(np.mean(diff > 0) * 100.0)
        under = float(np.mean(diff < 0) * 100.0)

        # Angular error (only where counts match)
        print("  Computing angular errors...")
        ang_err = np.full(pred_counts.shape, np.nan, np.float32)
        match_indices = np.where(diff == 0)[0]

        for i in match_indices:
            ang_err[i] = mean_ang_err_optimal(gt_dirs[i], pred_dirs[i])

        # Statistics for angular error
        ang_err_valid = ang_err[np.isfinite(ang_err)]
        if len(ang_err_valid) > 0:
            ang_err_mean = float(np.mean(ang_err_valid))
            ang_err_std = float(np.std(ang_err_valid))
        else:
            ang_err_mean = float("nan")
            ang_err_std = float("nan")

        # ========== PEAK-SPECIFIC STATISTICS ==========
        print("\n  Peak-Specific Statistics:")
        print(f"  {'Category':<15} {'N Voxels':<12} {'Accuracy %':<12} {'Ang Error (°)'}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*15}")

        peak_categories = [
            (1, "1 peak"),
            (2, "2 peaks"),
            (3, "3 peaks"),
            (None, ">3 peaks")  # None means "greater than 3"
        ]

        peak_stats = {}

        for peak_val, label in peak_categories:
            # Select voxels based on GT peak count
            if peak_val is None:  # >3 peaks
                mask = gt_counts > 3
            else:
                mask = gt_counts == peak_val

            n_vox_cat = mask.sum()

            if n_vox_cat == 0:
                print(f"  {label:<15} {0:<12} {'—':<12} {'—'}")
                peak_stats[label] = {
                    'n_voxels': 0,
                    'accuracy': None,
                    'angular_error_mean': None,
                    'angular_error_std': None,
                    'n_matching': 0
                }
                continue

            # Accuracy: percentage of exact matches in this category
            matches_cat = diff[mask] == 0
            acc_cat = float(matches_cat.mean() * 100.0)

            # Angular error: only for matching voxels in this category
            ang_err_cat = ang_err[mask]
            ang_err_cat_valid = ang_err_cat[np.isfinite(ang_err_cat)]

            if len(ang_err_cat_valid) > 0:
                ang_mean_cat = float(np.mean(ang_err_cat_valid))
                ang_std_cat = float(np.std(ang_err_cat_valid))
                ang_str = f"{ang_mean_cat:.2f} ± {ang_std_cat:.2f}"
            else:
                ang_mean_cat = float('nan')
                ang_std_cat = float('nan')
                ang_str = "—"

            print(f"  {label:<15} {n_vox_cat:<12,} {acc_cat:<12.2f} {ang_str}")

            # Store in dictionary
            peak_stats[label] = {
                'n_voxels': int(n_vox_cat),
                'accuracy': acc_cat,
                'angular_error_mean': ang_mean_cat,
                'angular_error_std': ang_std_cat,
                'n_matching': int(len(ang_err_cat_valid))
            }

        # Save arrays
        base = f"{self.ckpt_name}{self.suffix}"
        np.save(os.path.join(self.output_dir, f"fodf_peak_counts_pred_{base}.npy"), pred_counts)
        np.save(os.path.join(self.output_dir, f"fodf_peak_counts_gt_{base}.npy"), gt_counts)
        np.save(os.path.join(self.output_dir, f"fodf_peak_angerr_deg_{base}.npy"), ang_err)

        # Save peak-specific statistics to JSON
        peak_stats_path = os.path.join(self.output_dir, f"fodf_peak_specific_stats_{base}.json")
        with open(peak_stats_path, 'w') as f:
            json.dump(peak_stats, f, indent=4)
        print(f"\n  Peak-specific statistics saved to: {peak_stats_path}")

        # Save NIfTI maps
        print("  Saving NIfTI maps...")
        diff_map = np.full(wm.shape, np.nan, np.float32)
        diff_map[wm] = diff.astype(np.float32)
        save_nifti(
            diff_map, self.affine,
            os.path.join(self.output_dir, f"fodf_peak_count_diff_map_{base}.nii.gz"),
            "fODF Peak Count Difference Map (Pred - GT)"
        )

        ang_map = np.full(wm.shape, np.nan, np.float32)
        ang_map[wm] = ang_err
        save_nifti(
            ang_map, self.affine,
            os.path.join(self.output_dir, f"fodf_peak_angerr_deg_map_{base}.nii.gz"),
            "Mean angular error (deg) where counts match"
        )

        # Save category-specific error maps
        for peak_val, label in peak_categories:
            if peak_val is None:
                mask_cat = gt_counts > 3
                filename_suffix = "gt3peaks"
            else:
                mask_cat = gt_counts == peak_val
                filename_suffix = f"{peak_val}peak"

            if mask_cat.sum() == 0:
                continue

            # Create map for this category (difference)
            cat_diff_map = np.full(wm.shape, np.nan, np.float32)
            cat_diff_map[wm] = np.where(mask_cat, diff.astype(np.float32), np.nan)

            save_nifti(
                cat_diff_map, self.affine,
                os.path.join(self.output_dir, f"fodf_peak_diff_{filename_suffix}_{base}.nii.gz"),
                f"Peak count difference for {label} voxels"
            )

            # Create angular error map for this category
            cat_ang_map = np.full(wm.shape, np.nan, np.float32)
            cat_ang_map[wm] = np.where(mask_cat, ang_err, np.nan)

            save_nifti(
                cat_ang_map, self.affine,
                os.path.join(self.output_dir, f"fodf_peak_angerr_{filename_suffix}_{base}.nii.gz"),
                f"Angular error for {label} voxels where counts match"
            )

        # Peak distributions
        maxk = int(max(pred_counts.max(initial=0), gt_counts.max(initial=0)))
        pred_dist = {k: float(np.mean(pred_counts == k) * 100.0) for k in range(maxk + 1)}
        gt_dist = {k: float(np.mean(gt_counts == k) * 100.0) for k in range(maxk + 1)}

        # ========== FORMATTED OUTPUT ==========
        print(f"\nfODF Peak Count Results:")
        print(f"  N voxels: {n_vox:,}")
        print(f"  MAE: {mae:.2f} ± {mae_std:.2f} peaks")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  Over-estimated: {over:.2f}%")
        print(f"  Under-estimated: {under:.2f}%")
        print(f"  Angular error (when counts match): {ang_err_mean:.2f}° ± {ang_err_std:.2f}°")
        print(f"  N voxels with matching counts: {len(ang_err_valid):,}")

        # Distribution table
        print(f"\n  Peak Distribution:")
        print(f"  {'Peaks':<8} {'GT %':<10} {'Pred %':<10} {'Diff'}")
        print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

        # Show 1, 2, 3 peaks individually
        for k in [1, 2, 3]:
            gt_pct = gt_dist.get(k, 0.0)
            pred_pct = pred_dist.get(k, 0.0)

            diff_val = pred_pct - gt_pct
            diff_str = f"{diff_val:+.2f}" if abs(diff_val) > 0.01 else "—"
            print(f"  {k:<8} {gt_pct:>6.2f}%    {pred_pct:>6.2f}%    {diff_str}")

        # Summarize >3 peaks as "more"
        gt_more = sum(v for k, v in gt_dist.items() if k > 3)
        pred_more = sum(v for k, v in pred_dist.items() if k > 3)

        if gt_more > 0.01 or pred_more > 0.01:  # Only show if non-negligible
            diff_more = pred_more - gt_more
            diff_str = f"{diff_more:+.2f}" if abs(diff_more) > 0.01 else "—"
            print(f"  {'more':<8} {gt_more:>6.2f}%    {pred_more:>6.2f}%    {diff_str}")

        # Build result dictionary
        result = {
            "n_voxels": n_vox,
            "mae": mae,
            "mae_std": mae_std,
            "accuracy": acc,
            "over_estimated_pct": over,
            "under_estimated_pct": under,
            "angular_error_mean_deg": ang_err_mean,
            "angular_error_std_deg": ang_err_std,
            "n_matching_voxels": int(len(ang_err_valid)),
            "pred_distribution": pred_dist,
            "gt_distribution": gt_dist,
            "peak_specific_stats": peak_stats,
            "rel_peak_thresh": rel_peak_thresh,
            "abs_peak_thresh": abs_peak_thresh,
            "min_separation_angle_deg": min_sep_angle,
        }

        return result

    def save_image_quality_metrics(self, results: Dict, output_path: str = "image_quality_metrics.json"):
        """
        Save PSNR, SSIM, NRMSE, FSIM, ODF error, GFA error, and eigenvector error to JSON.

        Parameters
        ----------
        results : Dict
            Dictionary containing evaluation results with metrics
        output_path : str, optional
            Path where the JSON file will be saved (default: "image_quality_metrics.json")
        """
        # Build the output dictionary with metrics as top-level keys
        output_data = {}

        # Conventional image metrics - show FA and GFA for each metric
        metrics = ['PSNR', 'SSIM', 'NRMSE', 'FSIM']
        for metric in metrics:
            output_data[metric] = {}
            for map_name in ['FA', 'GFA']:
                key = f"{map_name.lower()}_{metric.lower()}"
                if key in results:
                    output_data[metric][map_name] = float(results[key])
                else:
                    output_data[metric][map_name] = None

        # Domain-specific metrics (single values, not per-map)
        if 'odf_error' in results:
            output_data['ODF_L2_Norm_Error_median'] = float(results['odf_error'])
        if 'gfa_error' in results:
            output_data['GFA_Absolute_Error_median'] = float(results['gfa_error'])
        if 'eigenvector_angular_error_median' in results:
            output_data['Eigenvector_Angular_Error_median_deg'] = float(results['eigenvector_angular_error_median'])

        # Save to JSON file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        print(f"\nImage quality metrics saved to: {output_path}")

        return output_path

    def print_summary(self, results: Dict):
        """Print a formatted summary of all evaluation metrics."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)

        print("\n--- Domain-Specific Metrics (WM-masked) ---")
        if 'odf_error' in results:
            print(f"  ODF L2-Norm Error (median, GFA-based WM): {results['odf_error']:.6f}")
        if 'gfa_error' in results:
            print(f"  GFA Absolute Error (median, GFA-based WM): {results['gfa_error']:.6f}")
        if 'eigenvector_angular_error_median' in results:
            print(f"  Eigenvector Angular Error (median, FA-based WM): {results['eigenvector_angular_error_median']:.2f}°")
            if 'eigenvector_angular_error_mean' in results:
                print(f"    Mean ± Std: {results['eigenvector_angular_error_mean']:.3f}° ± {results['eigenvector_angular_error_std']:.2f}°")
            if 'eigenvector_pct_above_30' in results:
                print(f"    Voxels with error >30°: {results['eigenvector_pct_above_30']:.2f}%")

        print("\n--- Conventional Image Metrics ---")

        # Standard order: PSNR, SSIM, NRMSE, FSIM - show FA and GFA for each
        metrics_info = [
            ('PSNR', '↑', ' dB'),
            ('SSIM', '↑', ''),
            ('NRMSE', '↓', ''),
            ('FSIM', '↑', ''),
        ]

        for metric_name, direction, unit in metrics_info:
            metric_lower = metric_name.lower()
            fa_key = f"fa_{metric_lower}"
            gfa_key = f"gfa_{metric_lower}"

            if fa_key in results or gfa_key in results:
                print(f"  {metric_name} {direction}:")
                if fa_key in results:
                    print(f"    FA:  {results[fa_key]:.4f}{unit}")
                if gfa_key in results:
                    print(f"    GFA: {results[gfa_key]:.4f}{unit}")


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


def main():
    # Get CLI arguments
    args = get_args()

    # Load checkpoint arguments and update them
    args = update_args(args)

    # Set default thresholds if not provided
    if not hasattr(args, 'gfa_wm_thresh'):
        args.gfa_wm_thresh = 0.20
    if not hasattr(args, 'fa_wm_thresh'):
        args.fa_wm_thresh = 0.20
    if not hasattr(args, 'keepalive_minutes'):
        args.keepalive_minutes = 5

    # Set a fixed seed for reproducibility
    set_seed(args.seed, reproducible=args.reproducible)

    # Set the precision for float32 matrix multiplication
    torch.set_float32_matmul_precision('medium')

    # Keep remote filesystem mounts active during long idle periods in processing
    interval = args.keepalive_minutes * 60
    stop_keepalive = keepalive([args.output_dir], interval=interval)

    # Initialize evaluation
    evaluation = Evaluation(args)

    # Store results for summary
    results = {}

    # Compute DTI
    compute_dti(
        data_dir=evaluation.args.data_dir,
        signal_path=evaluation.reconstructed_signal_path,
        mask_path=evaluation.mask_path,
        bval_path=evaluation.args.bval_path,
        bvec_path=evaluation.args.bvec_path,
        bval=evaluation.args.bval,
        tolerance=evaluation.args.tolerance,
        M=evaluation.args.M,
        suffix=evaluation.suffix,
        output_dir=evaluation.output_dir,
        ckpt_name=evaluation.ckpt_name,
        erosion_iter=None,
        fine_tune=evaluation.args.fine_tune,
        normed=True,
    )

    print("\n" + "="*80)
    print("COMPUTING EVALUATION METRICS")
    print("="*80)

    # Domain-specific metrics (WM-masked)
    print("\n--- Domain-Specific Metrics (WM-masked) ---")

    # ODF reconstruction error (GFA-based WM mask)
    odf_error_median, _ = evaluation.get_odf_error()
    results['odf_error'] = odf_error_median.item()

    # GFA error (GFA-based WM mask)
    gfa_error = evaluation.get_gfa_and_abs_error()
    if isinstance(gfa_error, float):
        results['gfa_error'] = gfa_error

    # Conventional image metrics
    print("\n--- Conventional Image Metrics ---")

    # PSNR (dB)
    (fa_psnr, _), (gfa_psnr, _) = evaluation.get_psnr()
    results.update({'fa_psnr': fa_psnr, 'gfa_psnr': gfa_psnr})

    # SSIM
    (fa_ssim, _), (gfa_ssim, _) = evaluation.get_ssim()
    results.update({'fa_ssim': fa_ssim, 'gfa_ssim': gfa_ssim})

    # FSIM
    (fa_fsim, _), (gfa_fsim, _) = evaluation.get_fsim()
    results.update({'fa_fsim': fa_fsim, 'gfa_fsim': gfa_fsim})

    # NRMSE (Normalized RMSE)
    fa_nrmse, gfa_nrmse = evaluation.get_nrmse()
    results.update({'fa_nrmse': fa_nrmse, 'gfa_nrmse': gfa_nrmse})

    # Directional preservation metrics
    print("\n--- Directional Preservation ---")

    # Eigenvector metrics (FA-based WM mask)
    eigenvec_results = evaluation.get_eigenvector_angular_error()
    if eigenvec_results is not None:
        results['eigenvector_angular_error_median'] = eigenvec_results['median']
        results['eigenvector_angular_error_mean'] = eigenvec_results['mean']
        results['eigenvector_angular_error_std'] = eigenvec_results['std']
        results['eigenvector_pct_above_30'] = eigenvec_results['pct_above_30']

    # Compute fODFs
    compute_fodfs(
        odf_sh_coeffs_path=evaluation.odf_sh_coeffs_path,
        mask_path=evaluation.mask_path,
        sh_order=evaluation.args.sh_order,
        output_path=evaluation.fodfs_path,
        roi=evaluation.args.roi
    )

    # fODF peak count metrics (GFA-based WM mask)
    fodf_eval = evaluation.get_fodf_peak_count_error()
    if fodf_eval is not None:
        results['fodf_peak_mae'] = fodf_eval['mae']
        results['fodf_peak_accuracy'] = fodf_eval['accuracy']
        results['fodf_peak_over_pct'] = fodf_eval['over_estimated_pct']
        results['fodf_peak_under_pct'] = fodf_eval['under_estimated_pct']
        results['fodf_peak_angular_error'] = fodf_eval['angular_error_mean_deg']

    # Print summary
    evaluation.print_summary(results)

    # Save metrics
    evaluation.save_image_quality_metrics(results, evaluation.cv_metrics_json_path)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

    # Compute tractography DIPY
    compute_tractography(
        fodfs_path=evaluation.fodfs_path,
        mask_path=evaluation.mask_path,
        fa_img_path=evaluation.fa_path,
        suffix=evaluation.suffix,
        ckpt_name=evaluation.ckpt_name,
        output_dir=evaluation.output_dir,
    )

    # Stop the keepalive thread
    stop_keepalive()


if __name__ == "__main__":
    main()
