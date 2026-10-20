import os
import argparse

import torch
import numpy as np
from dipy.align.reslice import reslice
from dipy.segment.mask import applymask

from utils import *


class SignalReconstructor:
    """Reconstruct dMRI signal from predicted ODF spherical harmonic coefficients."""
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.calcs_path = get_calcs_path(args.data_dir, args.fine_tune)
        self.ckpt_name, self.suffix = get_file_name_ckpt_suffix(args)

        print("\n==> Loading data...")

        # Load mask
        mask_path = self.args.mask_path
        if self.args.zooms:
            mask_path = os.path.join(self.calcs_path, f"mask{self.suffix}.nii.gz")
        self.mask_full = get_mask(mask_path, full_mask=True)[0]  # (X, Y, Z)

        # Load signal b0 mean
        self.signal_b0_mean, self.affine, self.zooms = load_nifti(self.args.b0_path)[:3]

        # Output directory
        experiment_version_dir = get_experiment_version_dir(self.args)
        self.output_path = os.path.join(experiment_version_dir, "reconstruction")
        os.makedirs(self.output_path, exist_ok=True)
        print(f"Output path: {self.output_path}\n")

        # Output paths
        self.output_signal_b0_mean_path = os.path.join(
            self.calcs_path,
            f'signal_b0_mean{self.suffix}.nii.gz'
        )
        self.recon_signal_path = os.path.join(
            self.output_path,
            f'recon_signal_without_b0_{self.ckpt_name}{self.suffix}.nii.gz'
        )
        self.recon_signal_expanded_path = os.path.join(
            self.output_path,
            f'recon_signal_{self.ckpt_name}{self.suffix}.nii.gz'
        )

        # Load ODF SH Coeffs
        self.odf_sh_coeffs_path = os.path.join(
            experiment_version_dir,
            "prediction",
            f'odf_sh_coeffs_{self.ckpt_name}{self.suffix}.pt'
        ) if not self.args.odf_sh_coeffs_path else self.args.odf_sh_coeffs_path

        print("==> Loading ODF SH Coeffs...")
        self.odf_sh_coeffs = get_odf_sh_coeffs(
            path=self.odf_sh_coeffs_path,
            mask=self.mask_full,
        ).to(self.args.device)  # (X, Y, Z, K)

    def _odf_sh_coeffs_to_signal(self) -> torch.Tensor:
        with Timer():
            # Reconstruct signal from ODF SH Coeffs
            odf_sh_coeffs_flat = self.odf_sh_coeffs[self.mask_full]  # (N, K)
            Phi_tensor = get_phi_r_tensors(self.args)[0]
            signal_pred = odf_sh_coeffs_flat @ Phi_tensor.T.to(self.args.device)  # (N, M)

            # Create signal tensor
            signal_recon = torch.zeros(
                (*self.odf_sh_coeffs.shape[:-1], signal_pred.shape[-1])
            ).to(self.args.device)  # (X, Y, Z, M)
            signal_recon[self.mask_full] = signal_pred
            print(f"Reconstructed signal of shape: {signal_recon.shape} without b0 volume.")

        return signal_recon

    def _add_b0_volume(self, signal_recon: np.ndarray) -> np.ndarray:
        with Timer():
            # Add b0 volume to the reconstructed signal
            signal_b0_mean = self.signal_b0_mean[..., np.newaxis]  # (X, Y, Z, 1)
            signal_recon_expanded = signal_recon * signal_b0_mean
            signal_recon_expanded = np.concatenate([signal_b0_mean, signal_recon_expanded], axis=-1)
            signal_recon_expanded = applymask(signal_recon_expanded, self.mask_full)  # (X, Y, Z, M + 1)

        return signal_recon_expanded

    def reconstruct(self, add_b0_volume: bool = False) -> None:
        # Reconstruct signal from ODF SH Coeffs
        signal_recon = self._odf_sh_coeffs_to_signal()  # X, Y, Z, M
        signal_recon = nan_to_num_tensor(signal_recon, nan=1e-6, posinf=1.0, neginf=1e-6)

        if self.args.verbose:
            print(f"Reconstructed signal min: {round(signal_recon.min().item(), 3)}")
            print(f"Reconstructed signal max: {round(signal_recon.max().item(), 3)}\n")

        # Clip to [0, 1]
        signal_recon = signal_recon.clip(0, 1).cpu().numpy()
        affine = self.affine

        if self.args.zooms:
            signal_b0_mean, affine = reslice(self.signal_b0_mean, self.affine, self.zooms, self.args.zooms)
            save_nifti(signal_b0_mean, affine, self.output_signal_b0_mean_path)

        # Save reconstructed signal without b0 volume
        save_nifti(signal_recon, affine, self.recon_signal_path)

        if add_b0_volume:
            # Add b0 volume and save reconstructed signal
            print("==> Adding b0 volume...")
            signal_recon_expanded = self._add_b0_volume(signal_recon)
            save_nifti(signal_recon_expanded, affine, self.recon_signal_expanded_path)


def main() -> None:
    """Set up and run reconstruction."""
    # Get CLI arguments
    args = get_args()

    # Load checkpoint arguments and update them
    args = update_args(args)

    # Set a fixed seed for reproducibility
    set_seed(args.seed, reproducible=args.reproducible)

    # Set the precision for float32 matrix multiplication
    torch.set_float32_matmul_precision('medium')

    # Reconstruct the signal
    signal_reconstructor = SignalReconstructor(args)
    signal_reconstructor.reconstruct()


if __name__ == "__main__":
    main()
