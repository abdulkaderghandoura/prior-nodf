import os
import argparse

import torch
import pytorch_lightning as pl

from models.nodf import NODF
from data.data_module import DataModule
from utils import *


def predict(args: argparse.Namespace) -> None:
    """Predict ODF spherical harmonic coefficients."""
    # Load model's weights from a checkpoint
    print(f"\n==> Loading model from:\n {args.ckpt_path}\n")
    model = NODF.load_from_checkpoint(
        args.ckpt_path,
        map_location="cpu",
        args=args,  # the checkpoint's arguments, updated with those passed explicitly (update_args)
    )

    # Get and create output directory
    experiment_version_dir = get_experiment_version_dir(args)
    output_dir = os.path.join(experiment_version_dir, "prediction")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize the data module and trainer
    data_module = DataModule(args)
    trainer = pl.Trainer(accelerator="auto", logger=False, devices=1)

    # Predict the ODF spherical harmonic coefficients
    print("\n==> Starting prediction...")
    with Timer():
        predictions = trainer.predict(model=model, datamodule=data_module)
        odf_sh_coeffs = torch.cat(predictions)
        print(f"Predicted {odf_sh_coeffs.shape[0]} points")

    # Save the ODF spherical harmonic coefficients
    ckpt_name, suffix = get_file_name_ckpt_suffix(args)
    output_path = os.path.join(output_dir, f'odf_sh_coeffs_{ckpt_name}{suffix}.pt')
    save_torch_tensor(odf_sh_coeffs.cpu().detach(), output_path, description="ODF SH coeffs")


def main() -> None:
    """Set up and run inference."""
    # Get CLI arguments
    args = get_args()

    # Load checkpoint arguments and update them
    args = update_args(args)

    # Set a fixed seed for reproducibility
    set_seed(args.seed, reproducible=args.reproducible)

    # Set the precision for float32 matrix multiplication
    torch.set_float32_matmul_precision('medium')

    # Predict ODF spherical harmonic coefficients
    predict(args)


if __name__ == "__main__":
    main()
