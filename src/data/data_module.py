import os
import argparse

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.dataset import ObservationPoints

from utils.transform import PointProcessor
from utils.utility import (
    get_mask,
    reslice_mask,
    get_file_name_ckpt_suffix,
    create_grid,
    create_voxel_subgrid,
    get_calcs_path,
    load_nifti,
    save_torch_tensor,
    normalize_signal,
)


class DataModule(pl.LightningDataModule):
    """Data module for the observation points"""
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.calcs_dir = get_calcs_path(args.data_dir, args.fine_tune)
        print("\n==> Initializing data module...")

        # Initialize the point processor
        self.processor = PointProcessor(
            moving_path=self.args.moving,
            fixed_path=self.args.fixed if self.args.fine_tune else None,
            transform_path=self.args.transform if self.args.fine_tune else None,
            device=self.args.device if torch.cuda.device_count() == 1 else "cpu"
        )

        # Get the min and max physical values of the moving image
        self.min_vals, self.max_vals = self.processor.get_min_max_moving_physical_vals(
            continuous=False,
            verbose=self.args.verbose,
        )

        # Get the file name suffix
        self.suffix = get_file_name_ckpt_suffix(self.args)[1]

    def setup(self, stage: str = None) -> None:
        # Load mask
        mask = get_mask(self.args.mask_path, full_mask=True)[0]

        # Helper variables
        parts = os.path.dirname(self.args.moving).split('/')
        moving_name = "_".join([parts[parts.index("data") + 1]] + parts[parts.index("processed") + 1:])
        transform_label = f"_transformed_to_{moving_name}" if self.args.fine_tune else ""

        # Paths to the computed coordinates, masks, and affine matrix
        coords_file_name = f'physical_coords{transform_label}{self.suffix}.pt'
        aux_coords_file_name = f'aux_physical_coords{transform_label}{self.suffix}.pt'
        coords_path = os.path.join(self.calcs_dir, coords_file_name)
        aux_coords_path = os.path.join(self.calcs_dir, aux_coords_file_name)
        resliced_mask_path = os.path.join(self.calcs_dir, f"mask{self.suffix}.nii.gz")

        aux_coords = None  # Subgrid of each voxel
        if os.path.exists(coords_path) and not self.args.preprocess:
            # Load precomputed coords
            print(f"Using precomputed coords from {coords_path}")
            coords = torch.load(coords_path)
            if stage == "fit" and self.args.downsampling_subgrid_size:
                # Load precomputed auxiliary coords
                print(f"Using precomputed aux coords from {aux_coords_path}")
                aux_coords = torch.load(aux_coords_path)

        else:
            # Get mask for the coordinates
            coords_mask = mask

            # Get coordinates
            step = [1, 1, 1]

            if self.args.zooms:
                # Create and save a resliced mask
                coords_mask = reslice_mask(
                    input_path=self.args.mask_path,
                    new_zooms=self.args.zooms,
                    reslice_like=self.args.reslice_like,
                    mask_output_path=resliced_mask_path
                )

                assert self.args.continuous, "Re-run with the --continuous flag to generate continuous coordinates"
                original_zooms = load_nifti(self.args.mask_path, verbose=False)[2]
                step = [self.args.zooms[i] / original_zooms[i] for i in range(3)]
                print(f"np.arange step: {step}")

            indices = create_grid(
                size=mask.shape,
                step=step,
                mask=coords_mask,
                verbose=True,
            )

            physical_points = self.processor.index2physical(
                index_points=indices,
                from_fixed=self.args.fine_tune,
                continuous=self.args.continuous
            )

            if self.args.fine_tune:
                physical_points = self.processor.transform_physical_points(physical_points)

            if stage == "fit" and self.args.downsampling_subgrid_size:
                # Add auxiliary points for the downsampling operator in the loss computation
                aux_indices = create_voxel_subgrid(
                    centers=indices,
                    size=self.args.downsampling_subgrid_size,
                ).view(-1, 3)  # NM x 3

                aux_physical_points = self.processor.index2physical(
                    index_points=aux_indices,
                    from_fixed=self.args.fine_tune,
                    continuous=True  # Always continuous
                )

                if self.args.fine_tune:
                    aux_physical_points = self.processor.transform_physical_points(aux_physical_points)

                # Concatenate temporarily for normalization
                physical_points = torch.cat([physical_points, aux_physical_points], dim=0)

            physical_points = self.processor.clamp_points(physical_points, (self.min_vals, self.max_vals))
            coords = (physical_points - self.min_vals) / (self.max_vals - self.min_vals)

            if stage == "fit" and self.args.downsampling_subgrid_size:
                aux_coords = coords[-len(aux_physical_points):].clone()
                coords = coords[:-len(aux_physical_points)].clone()

                assert coords.size() == indices.size()
                aux_coords = aux_coords.view(indices.size(0), -1, 3)
                save_torch_tensor(aux_coords, aux_coords_path, description="auxiliary coords")

            save_torch_tensor(coords, coords_path, description="coords")

        # Send coords and aux_coords to the CPU
        coords = coords.to('cpu')
        if aux_coords is not None:
            aux_coords = aux_coords.to('cpu')

        if stage == "fit":
            # Check if aux_coords are generated
            if self.args.downsampling_subgrid_size:
                assert aux_coords is not None, "Re-run with the --preprocess flag to generate aux_coords"

            # Load and normalize signal
            signal = load_nifti(self.args.signal_path)[0]
            signal_normed = normalize_signal(
                signal=signal,
                mask=mask,
                data_dir=self.args.data_dir,
                bval=self.args.bval,
                tolerance=self.args.tolerance,
            )
            signal_normed = signal_normed[mask]  # (N, M)
            print(f"Signal shape: {signal_normed.shape}")

            # Create dataset
            self.dataset = ObservationPoints(coords, signal_normed, aux_coords)

        if stage == "predict":
            # Create dataset
            self.dataset = ObservationPoints(coords)

        print(f"Coords shape: {self.dataset.coords.shape}")

    def train_dataloader(self) -> DataLoader:
        print(f"Dataset length (train): {len(self.dataset)}\n")
        return DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            pin_memory=self.args.pin_memory,
            num_workers=self.args.num_workers,
            drop_last=(self.args.batch_size < len(self.dataset)),
        )

    def predict_dataloader(self) -> DataLoader:
        print(f"Dataset length (predict): {len(self.dataset)}\n")
        return DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            pin_memory=self.args.pin_memory,
            num_workers=self.args.num_workers,
            drop_last=(self.args.batch_size < len(self.dataset)),
        )
