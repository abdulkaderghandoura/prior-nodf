import os
import sys
import time
import logging
from typing import List

from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utility import Timer

logger = logging.getLogger(__name__)


class CustomCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that also saves a snapshot at each epoch in `custom_epochs`."""
    def __init__(self, custom_epochs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_epochs = set(custom_epochs)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        # Save checkpoints as usual
        super().on_train_epoch_end(trainer, pl_module)

        # Save snapshots for custom epochs
        if trainer.current_epoch in self.custom_epochs:
            ckpt_name = f"epoch_{trainer.current_epoch}_snapshot.ckpt"
            ckpt_path = os.path.join(self.dirpath, ckpt_name)
            try:
                self._save_checkpoint(trainer, ckpt_path)
                print(f"Saved checkpoint/snapshot for epoch {trainer.current_epoch} "
                      f"with args.weights_only={self.save_weights_only}\n")
            except Exception:
                logger.error("Checkpoint save failed", exc_info=True)


class WallClockCallback(ModelCheckpoint):
    """
    Save checkpoints at given elapsed minutes since training start.
    Prints and logs: "{minute} minutes\t{epoch} epochs\t{ckpt_path}"
    Also appends the same line to a mapping file.
    """
    def __init__(
        self,
        minutes_list: List[int],
        timer: Timer = None,
        mapping_filename="time_checkpoints.txt",
        *args,
        **kwargs
    ):
        super().__init__(
            *args,
            save_on_train_epoch_end=False,
            save_last=False,
            every_n_epochs=0,
            every_n_train_steps=0,
            monitor=None,
            **kwargs
        )
        self.timer = timer
        self.targets = sorted(int(m) for m in minutes_list)
        self.targets_sec = [m * 60 for m in self.targets]
        self._next_idx = 0
        self.mapping_filename = mapping_filename
        self._mapping_path = None

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        # Lazily create/start timer to avoid pre-fit and multi-process issues
        if self.timer is None:
            self.timer = Timer()
        if getattr(self.timer, "start_time", None) is None:
            self.timer.start()

        # Resolve save dir and mapping file
        self.dirpath = self.dirpath or getattr(trainer.checkpoint_callback, "dirpath", None) or trainer.default_root_dir
        os.makedirs(self.dirpath, exist_ok=True)
        self._mapping_path = os.path.join(self.dirpath, self.mapping_filename)

    def _maybe_save(self, trainer):
        if self._next_idx >= len(self.targets_sec):
            return
        elapsed_sec = int(self.timer.elapsed())
        while self._next_idx < len(self.targets_sec) and elapsed_sec >= self.targets_sec[self._next_idx]:
            minute = self.targets[self._next_idx]
            epoch = int(trainer.current_epoch)
            ckpt_name = f"time_{minute:05d}min_epoch{trainer.current_epoch}.ckpt"
            ckpt_path = os.path.join(self.dirpath, ckpt_name)
            self._save_checkpoint(trainer, ckpt_path)

            line = f"{minute} minutes\t{epoch} epochs\t{ckpt_path}"
            # Only main process should print or write files in multi-GPU/DDP runs
            if trainer.is_global_zero:
                # print to log/console
                rank_zero_info(line)
                # append to mapping file
                with open(self._mapping_path, "a") as f:
                    f.write(line + "\n")
                # optional: write to TensorBoard text if available
                exp = getattr(trainer.logger, "experiment", None)
                if exp is not None and hasattr(exp, "add_text"):
                    exp.add_text("time_checkpoints", line, global_step=minute)

            self._next_idx += 1

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._maybe_save(trainer)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        self._maybe_save(trainer)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        return  # disabled to avoid epoch-end autosave


class CustomLogger(Callback):
    """Log metrics at the end of each training epoch."""
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in trainer.callback_metrics.items()
            if k.endswith("/train")
        }

        trainer.logger.log_metrics(metrics, step=trainer.current_epoch)


class TimeLoggingCallback(Callback):
    """Log the time taken for each epoch."""
    def setup(self, trainer, pl_module, stage=None):
        self.custom_logger = logging.getLogger("pytorch_lightning")
        self.training_start_time_memory = None
        self.validation_start_time_memory = None
        self.validation_epoch_duration = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.training_start_time_memory = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - self.training_start_time_memory - self.validation_epoch_duration
        self.custom_logger.info(f" Epoch training time: {epoch_duration:.2f}s")
        self.training_start_time_memory = None
        self.validation_epoch_duration = 0

    def on_validation_epoch_start(self, trainer, pl_module):
        self.validation_start_time_memory = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        self.validation_epoch_duration = time.time() - self.validation_start_time_memory
        self.custom_logger.info(f" Epoch validation time: {self.validation_epoch_duration:.2f}s")
        self.validation_start_time_memory = None
