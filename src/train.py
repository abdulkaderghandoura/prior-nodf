import os
import argparse
import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler

from models.nodf import NODF
from data.data_module import DataModule
from utils.utility import *
from utils.custom_callbacks import *


def init_epoch_checkpoint_callback(args, output_dir: str) -> CustomCheckpoint:
    """Initialize the checkpoint callback for the trainer."""
    # Save last checkpoint if not multiple of every_n_epochs
    snapshot_epochs = list(args.snapshot_epochs)
    if args.num_epochs % args.every_n_epochs != 0:
        snapshot_epochs.append(args.num_epochs - 1)

    # Initialize the checkpoint callback
    checkpoint_callback = CustomCheckpoint(
        custom_epochs=snapshot_epochs,
        dirpath=output_dir,
        filename="epoch_{epoch}-step_{step}",
        save_top_k=-1,  # all instead of top k
        every_n_epochs=args.every_n_epochs,
        save_weights_only=args.weights_only,
        auto_insert_metric_name=False,
    )

    return checkpoint_callback


def init_time_checkpoint_callback(args, output_dir: str) -> WallClockCallback:
    """Initialize the time-based checkpoint callback for the trainer."""
    return WallClockCallback(
        minutes_list=args.snapshot_minutes,
        dirpath=output_dir,
        save_top_k=-1,
        save_weights_only=args.weights_only,
        filename=None,
        auto_insert_metric_name=False,
    )


def initialize_model(args) -> pl.LightningModule:
    """Initialize the model for training or fine-tuning."""
    # Initialize the model from scratch
    if not args.ckpt_path:
        return NODF(args)

    # Load model weights for fine-tuning
    if args.fine_tune:
        model = NODF(args)
        print(f"==> Loading model weights from {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return model

    # Continue training from a checkpoint
    print("==> Continuing training from a checkpoint...")
    return NODF.load_from_checkpoint(
        args.ckpt_path,
        map_location="cpu",
    )


def train(args: argparse.Namespace) -> None:
    """Train the model."""
    # Initialize the data module, logger, and callbacks
    data_module = DataModule(args)
    tensorboard_logger = None
    profiler = None
    callbacks = [TimeLoggingCallback()]

    # Regular runs or debug WITH TensorBoard
    if not args.debug:
        print("==> Initializing TensorBoard logger...")
        logger_path = os.path.join(args.log_dir, 'debug' if args.debug_with_tensorboard else '', args.experiment_name)
        tensorboard_logger = TensorBoardLogger(save_dir=logger_path, name=args.model, default_hp_metric=True)
        print(f'Logger path: {os.path.join(logger_path, args.model, "version_" + str(tensorboard_logger.version))}')
        callbacks.append(CustomLogger())

    # Regular runs only
    if not args.debug and not args.debug_with_tensorboard:
        print("==> Initializing checkpoint callback...")
        # Create output path
        args.experiment_version = tensorboard_logger.version
        experiment_version_dir = get_experiment_version_dir(args)
        output_dir = os.path.join(experiment_version_dir, "training")
        print(f"Output path: {output_dir}")

        # Initialize the checkpoint callbacks
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        callbacks.append(init_epoch_checkpoint_callback(args, checkpoint_dir))
        callbacks.append(
            init_time_checkpoint_callback(
                args, output_dir=checkpoint_dir
            )
        )

        # Write ERROR-level log records, such as a failed snapshot save, to errors.log
        logging.basicConfig(
            filename=os.path.join(output_dir, 'errors.log'),
            level=logging.ERROR,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Initialize a simple profiler if requested
        if args.profiler:
            print("==> Initializing profiler...")
            profiler_dir_path = os.path.join(output_dir, "profiler")
            profiler = SimpleProfiler(dirpath=profiler_dir_path, filename="profile_report")

    # Initialize the trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=tensorboard_logger,
        accelerator="auto",
        max_epochs=args.num_epochs,
        devices="auto",
        profiler=profiler if profiler else None,
        gradient_clip_val=args.gradient_clip_val,
        enable_checkpointing=(not args.debug and not args.debug_with_tensorboard)
    )

    # Initialize the model and start training
    print("\n==> Initializing model and starting training...")
    model = initialize_model(args=args)
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=None if args.fine_tune else args.ckpt_path
    )

    # Save checkpoints in regular runs only
    if not args.debug and not args.debug_with_tensorboard:
        print("Training completed! Checkpoints saved to:", "\n", checkpoint_dir)


def main() -> None:
    """Set up and run training."""
    # Get CLI arguments
    args = get_args()

    # Set a fixed seed for reproducibility
    set_seed(args.seed, reproducible=args.reproducible)

    # Set the precision for float32 matrix multiplication
    torch.set_float32_matmul_precision('medium')

    # Train the model
    train(args)


if __name__ == "__main__":
    main()
