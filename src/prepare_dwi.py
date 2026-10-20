"""
Prepare raw diffusion MRI data for the pipeline.

Format conditioning only, not signal processing. For each subject, the raw DWI becomes the
layout the rest of the code reads, under
<workspace>/<environment>/data/<dataset>/processed/sub-<id>/<HighRes|LowRes>/:

    signal.nii.gz   the DWI, re-stored with consistent strides
    bval.txt        b-values, FSL format
    bvec.txt        b-vectors, FSL format, matching the image orientation
    b0.nii.gz       mean of the b=0 volumes
    mask.nii.gz     brain mask (uint8): the provided one, or dwi2mask's
    train_calcs/, finetune_calcs/   empty folders the pipeline caches into

Nothing is denoised or corrected. The HCP data used in the paper had already been through
the HCP's own minimal preprocessing pipelines upstream; this script only reformats it.
Requires MRtrix3.
"""
import os
import time
import argparse
import subprocess
import shutil
import tempfile
import getpass
import platform
import logging
import re
from pathlib import Path
from typing import List, Union, Dict, Any

import yaml
import nibabel as nib
import numpy as np
from dipy.io.gradients import read_bvals_bvecs

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

# Module-specific logger
logger = logging.getLogger(__name__)


class DWIPreparer:
    """Prepare raw dMRI data for the pipeline with MRtrix3 (see the module docstring)."""
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.resolution = "HighRes" if self.args.resolution == "high" else "LowRes"

        # Initialize subject-specific variables
        self.dtype: Dict[str, str] = {}
        self.current_dwi: Dict[str, Path] = {}
        self.processed_sub_dir: Dict[str, Path] = {}
        self.tmp_dir: Dict[str, Path] = {}
        self.subject_paths: Dict[str, Dict[str, Path]] = {}

        # Get config and necessary paths (preserves execution order)
        self.config = self._load_config()
        self.workspace_dir = self._get_workspace_dir()
        self.output_base_dir = self._get_output_base_dir()

        # Check if user has write access to self.workspace_dir
        self._run_sanity_check()

        # Check if the required tools are installed
        self._check_dependencies()

        # Get temporary base directory
        self.tmp_base = self._get_tmp_base()

        # Set the number of threads for MRtrix3 tools
        os.environ.setdefault("MRTRIX_NTHREADS", str(self.args.n_threads))

    def _load_config(self) -> Dict[str, Any]:
        """Load the config file."""
        # Check if a config file is provided
        if self.args.config is not None:
            config_path = Path(self.args.config)
            # Ensure provided config file exists and is a regular file
            if not config_path.is_file():
                raise FileNotFoundError(f"Config file {config_path} does not exist.")
        else:
            # Get default config file if not provided (adjust if needed)
            config_path = Path(__file__).parent.parent / "configs" / "datasets.yaml"
            # Require --config if default file is missing
            if not config_path.is_file():
                raise ValueError("Config file is required. Please set --config.")

        # Read config file
        self.config_path = config_path.resolve()
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _get_workspace_dir(self) -> Path:
        """Get the workspace directory."""
        # Check if a workspace is provided
        if self.args.personal_workspace is not None:
            workspace_path = Path(self.args.personal_workspace)
            # Ensure the workspace exists and is a directory
            if not workspace_path.is_dir():
                raise FileNotFoundError(
                    f"Workspace directory {workspace_path} does not exist."
                )
        else:
            # Set workspace to default if not provided
            workspace_path = self.config["environment"]["workspace_dir"][self._get_platform()]
            if workspace_path is not None:
                workspace_path = Path(workspace_path.format(user=self._get_user()))
                if not workspace_path.is_dir():
                    raise FileNotFoundError(
                        f"Workspace directory {workspace_path} does not exist."
                    )
            else:
                raise FileNotFoundError(
                    "Workspace directory is not set. Please update the config "
                    f"file {self.config_path} or set --personal_workspace.")

        return workspace_path.resolve()

    def _get_output_base_dir(self) -> Path:
        """Get the output base directory."""
        return self.workspace_dir / self.config["environment"]["name"] / "data"

    def _check_dependencies(self) -> None:
        """Check that the MRtrix3 tools this script calls are installed."""
        needed = ["mrinfo", "mrconvert", "dwiextract", "mrmath", "dwi2mask"]
        missing = [t for t in needed if shutil.which(t) is None]
        if missing:
            raise FileNotFoundError(f"Missing MRtrix3 tools: {missing}")

    def _get_platform(self) -> str:
        """Get the platform of the system."""
        platform_name = platform.system()
        if platform_name in {"Linux", "Darwin"}:
            return "POSIX"
        elif platform_name == "Windows":
            return "Windows"
        else:
            raise ValueError(f"Unsupported platform: {platform_name}")

    def _get_user(self) -> str:
        """Get the user of the system."""
        return getpass.getuser()

    def _run_sanity_check(self) -> None:
        """
        Check if the user has write access to the workspace directory.
        This helper is to avoid unintentionally overwriting data of other users.
        """
        if not os.access(self.workspace_dir, os.W_OK):
            raise PermissionError(f"Access denied: workspace '{self.workspace_dir}'")

    def _get_tmp_base(self) -> Path:
        """
        Get the temporary base directory.
        Use temporary directory with fast I/O on local storage:
        SLURM_TMPDIR for clusters if set, else fallback to system temp
        """
        return Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir())).resolve()

    def _get_subject_paths(self, sub_id: str) -> Dict[str, Path]:
        """Set raw and processed data paths for a subject with a given ID."""
        # Get data configuration
        data_config = self.config["data"][self.args.data_name][self.resolution]
        source_dir = data_config["source_dir"].format(id=sub_id)

        # Get data source directory
        if source_dir is None or not Path(source_dir).is_dir():
            raise FileNotFoundError(f"Source directory {source_dir} not found.")
        data_source_dir = Path(source_dir).resolve()

        # Set required paths
        subject_paths = {
            "raw_signal": data_source_dir / data_config["dwi"].format(id=sub_id),
            "raw_bval": data_source_dir / data_config["bval"].format(id=sub_id),
            "raw_bvec": data_source_dir / data_config["bvec"].format(id=sub_id),
        }

        # Check if the required files are present
        missing = [k for k, p in subject_paths.items() if not p.is_file()]
        if missing:
            details = ", ".join(f"{k}={subject_paths[k]}" for k in missing)
            raise FileNotFoundError(f"Missing required files for sub-{sub_id}: {details}")

        if data_config["mask"] is not None:
            subject_paths["raw_mask"] = data_source_dir / data_config["mask"].format(id=sub_id)

        if data_config["json"] is not None:
            subject_paths["json"] = data_source_dir / data_config["json"].format(id=sub_id)

        return subject_paths

    def _is_relative_to(self, dir: Path, base: Path) -> bool:
        """Equivalent to Path.is_relative_to() in Python 3.9+."""
        try:
            dir.relative_to(base)
            return True
        except ValueError:
            return False

    def _safe_rmtree(self, dir: Path, base: Path) -> None:
        """Remove a directory only if it is within the base directory."""
        abs_dir = Path(dir).resolve()
        abs_base_dir = Path(base).resolve()

        # Check if the base directory is relative to the tmp base or the personal workspace
        if not self._is_relative_to(dir=abs_base_dir, base=self.tmp_base) and \
            not self._is_relative_to(dir=abs_base_dir, base=self.workspace_dir):
            raise ValueError(
                f"The script assumes all output paths start with:\n {self.workspace_dir}"
                "Please set --personal_workspace to a directory you own."
            )

        # CAUTION: use rmtree very carefully!
        if not self._is_relative_to(dir=abs_dir, base=abs_base_dir):
            raise ValueError(f"Refusing to delete outside base: {abs_dir}")

        # Remove the directory
        shutil.rmtree(abs_dir)

    def _save_raw_data(self, sub_id: str) -> None:
        """Save a copy of the raw data to the output directory."""
        with Timer():
            # Get raw data directory path
            raw_data_dir = self.output_base_dir / self.args.data_name / "raw"
            raw_sub_dir = raw_data_dir / f"sub-{sub_id}" / self.resolution

            # Check if the raw data directory exists and is allowed to be overwritten
            if raw_sub_dir.is_dir() and self.args.overwrite:
                self._safe_rmtree(raw_sub_dir, self.workspace_dir)

            # Create raw data directory
            raw_sub_dir.mkdir(parents=True, exist_ok=True)
            raw_paths = self.subject_paths[sub_id]

            # Backup raw data to the raw data directory
            shutil.copy(raw_paths["raw_signal"], raw_sub_dir / raw_paths["raw_signal"].name)
            shutil.copy(raw_paths["raw_bval"], raw_sub_dir / raw_paths["raw_bval"].name)
            shutil.copy(raw_paths["raw_bvec"], raw_sub_dir / raw_paths["raw_bvec"].name)

            if "raw_mask" in raw_paths:
                shutil.copy(raw_paths["raw_mask"], raw_sub_dir / raw_paths["raw_mask"].name)

            if "json" in raw_paths:
                shutil.copy(raw_paths["json"], raw_sub_dir / raw_paths["json"].name)

            logger.info(f"Raw data backup saved to:\n {raw_sub_dir}")

    def init_subject(self, sub_id: str) -> None:
        """Initialize the subject."""
        self.subject_paths[sub_id] = self._get_subject_paths(sub_id)
        raw_signal_path = self.subject_paths[sub_id]["raw_signal"]
        self.dtype[sub_id] = nib.load(raw_signal_path).get_data_dtype().name
        self.current_dwi[sub_id] = raw_signal_path

        # Create processed output directory
        processed_data_dir = self.output_base_dir / self.args.data_name / "processed"
        self.processed_sub_dir[sub_id] = processed_data_dir / f"sub-{sub_id}" / self.resolution

        # Check if processed subject directory exists and is allowed to be overwritten
        if self.processed_sub_dir[sub_id].is_dir():
            if self.args.overwrite:
                self._safe_rmtree(self.processed_sub_dir[sub_id], self.workspace_dir)
            else:
                raise FileExistsError(
                    f"Processed subject output directory {self.processed_sub_dir[sub_id]} "
                    "already exists. Use --overwrite to overwrite."
                )

        # Create output subdirectories
        (self.processed_sub_dir[sub_id] / "train_calcs").mkdir(parents=True, exist_ok=True)
        (self.processed_sub_dir[sub_id] / "finetune_calcs").mkdir(parents=True, exist_ok=True)

        # Backup raw data if requested
        if self.args.save_raw_data:
            self._save_raw_data(sub_id)

        # Strip root like '/' to prevent overriding temp base path
        assert self.processed_sub_dir[sub_id].is_absolute(), \
            "Processed subject directory must be absolute."
        rel_proc_sub_dir = self.processed_sub_dir[sub_id].relative_to(self.workspace_dir)

        # Create tmp directory in a local storage for faster I/O operations
        self.tmp_dir[sub_id] = self.tmp_base / self._get_user() / ".cache" / rel_proc_sub_dir

    def _run_command(self, cmd_list: List[Union[str, Path]]) -> None:
        """Run a shell command and print it."""
        str_cmd_list = list(map(str, cmd_list))
        if self.args.verbose:
            logger.info(f"Running: {' '.join(str_cmd_list)}")
        subprocess.run(str_cmd_list, check=True)

    def _has_negative_x_stride(self, mif_path):
        """Whether the image is stored with a negative x stride (MRtrix strides -1,2,3,4)."""
        data_info = subprocess.check_output(['mrinfo', mif_path]).decode()
        matched = re.search(r'Data strides:\s*\[([^\]]+)\]', data_info)
        if not matched:
            raise ValueError("Cannot find strides")
        strides = [int(x) for x in matched.group(1).split()]
        if strides == [-1, 2, 3, 4]:
            return True
        if strides == [1, 2, 3, 4]:
            return False

        raise NotImplementedError(f"Unexpected strides: {strides}")

    def _convert_dwi_to_mif(self, tmp_dir: Path, sub_id: str, dtype: str) -> None:
        """Convert the raw signal and gradients into MRtrix3 format."""
        # Use -stride 2,3,4,1 for volume-contiguous storage to enable
        # faster voxel-wise access and lower RAM usage in MRtrix3 dMRI tools
        with Timer():
            bvec_path = self.subject_paths[sub_id]["raw_bvec"]
            if self._has_negative_x_stride(self.current_dwi[sub_id]):
                print("Flipping x component of b-vectors due to negative x stride...")
                bvecs = read_bvals_bvecs(fbvals=None, fbvecs=str(bvec_path))[1]
                bvecs[:, 0] *= -1  # flip x component (column 0)
                bvec_path = tmp_dir / 'bvec_flipped.txt'
                np.savetxt(bvec_path, bvecs, fmt='%.16f')

            out_dwi = tmp_dir / 'dwi.mif'
            self._run_command([
                "mrconvert", self.current_dwi[sub_id], out_dwi,
                "-fslgrad", bvec_path, self.subject_paths[sub_id]["raw_bval"],
                "-datatype", dtype, "-stride", "2,3,4,1"
            ])
            self.current_dwi[sub_id] = out_dwi

    def _export_nifti_and_gradients(self, tmp_dir: Path, sub_id: str, dtype: str) -> None:
        """
        Export the signal to NIfTI format and gradients to FSL format.
        NIfTI cannot store arbitrary strides, so MRtrix reorders strides to [1, 2, 3, 4].
        """
        with Timer():
            self._run_command([
                "mrconvert", self.current_dwi[sub_id], tmp_dir / 'signal.nii.gz',
                "-datatype", dtype, "-export_grad_fsl",
                tmp_dir / 'bvec.txt', tmp_dir / 'bval.txt'
            ])

    def _average_b0_volumes(self, tmp_dir: Path, sub_id: str, dtype: str) -> None:
        """Average the b0 volumes and export the result to NIfTI format."""
        with Timer():
            self._run_command([
                "dwiextract", self.current_dwi[sub_id],
                "-bzero", tmp_dir / 'b0_volumes.mif'
            ])
            self._run_command([
                "mrmath", tmp_dir / 'b0_volumes.mif',
                "mean", "-axis", "3", tmp_dir / 'b0_mean.mif'
            ])
            self._run_command([
                "mrconvert", tmp_dir / 'b0_mean.mif',
                tmp_dir / 'b0.nii.gz', "-datatype", dtype
            ])

    def _generate_brain_mask(self, tmp_dir: Path, sub_id: str) -> None:
        """Generate the brain mask from the DWI data."""
        with Timer():
            self._run_command([
                "dwi2mask", self.current_dwi[sub_id], tmp_dir / 'mask_dwi2mask.mif'
            ])

    def _convert_and_export_mask(self, input_mask: Path, tmp_dir: Path) -> None:
        """
        Convert the mask into MRtrix3 format with bit datatype and consistent strides
        for subsequent processing, and export it to NIfTI format with uint8 datatype.
        """
        with Timer():
            self._run_command([
                "mrconvert", input_mask, tmp_dir / 'mask.mif',
                "-datatype", "bit", "-stride", "1,2,3"
            ])
            self._run_command([
                "mrconvert", tmp_dir / 'mask.mif',
                tmp_dir / 'mask.nii.gz', "-datatype", "uint8"
            ])

    def run_pipeline(self, sub_id: str) -> None:
        """Convert one subject's raw DWI into the layout the pipeline reads."""
        # Work in a temporary directory on local storage for faster I/O
        tmp = self.tmp_dir[sub_id] / f"run_{os.getpid()}"
        tmp.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"\n=== PREPARING SUBJECT {sub_id} ===\n")

            # Convert the raw signal and gradients into dwi.mif with -stride 2,3,4,1
            logger.info("==> Converting raw data into MRtrix3 format...")
            self._convert_dwi_to_mif(tmp, sub_id, dtype=self.dtype[sub_id])

            # Export the adjusted signal to signal.nii.gz and gradients to bvec.txt and bval.txt
            logger.info("==> Exporting dwi to NIfTI and gradients to FSL...")
            self._export_nifti_and_gradients(tmp, sub_id, dtype=self.dtype[sub_id])

            # Extract the b0 volumes and average them to b0.nii.gz
            logger.info("==> Averaging b0 volumes and exporting to NIfTI format...")
            self._average_b0_volumes(tmp, sub_id, dtype=self.dtype[sub_id])

            # Generate a mask if the raw one is not provided
            if "raw_mask" not in self.subject_paths[sub_id]:
                logger.info("==> Generating brain mask from DWI data...")
                self._generate_brain_mask(tmp, sub_id)
                self._convert_and_export_mask(tmp / 'mask_dwi2mask.mif', tmp)
            else:
                logger.info("==> Converting raw mask into MRtrix3 format...")
                self._convert_and_export_mask(self.subject_paths[sub_id]["raw_mask"], tmp)

            # Move the prepared data to the output directory
            logger.info("==> Moving the prepared data to the output directory...")
            with Timer():
                for name in ['signal.nii.gz', 'mask.nii.gz', 'bvec.txt', 'bval.txt', 'b0.nii.gz']:
                    # str(): shutil.move rejects Path sources when the target is a directory before Python 3.9
                    shutil.move(str(tmp / name), str(self.processed_sub_dir[sub_id]))

            logger.info(f"Processed subject output directory: {self.processed_sub_dir[sub_id]}")
            logger.info(f"✔  Prepared subject {sub_id}.")

            # Final cleanup
            logger.info("==> Removing temporary files...")
            with Timer():
                self._safe_rmtree(tmp, self.tmp_base)

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user (Ctrl+C).")
            raise  # Don't suppress or wrap with an exception handler

        except Exception as e:
            msg = f"Error while preparing subject {sub_id}: {e}"
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from e

        finally:
            logger.info("==> Final cleanup of temporary files...")
            with Timer():
                # Cleanup main tmp directory
                for p in list(self.tmp_dir.values()):
                    if p.exists():
                        try:
                            self._safe_rmtree(p, self.tmp_base)
                        except Exception as e:
                            msg = f"Failed to remove the directory {p}: {e}"
                            logger.error(msg, exc_info=True)
                            raise RuntimeError(msg) from e

        logger.info(f"✔  Output files moved and caches cleaned for subject {sub_id}.")


class Timer:
    """Class for measuring execution time."""
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def report(self):
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")

        end_time = time.perf_counter()
        elapsed = end_time - self.start_time
        logger.info(f"Step executed in {(elapsed * 1000):.4f} ms ({(elapsed):.4f} s)\n")

        return elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.report()


def main() -> None:
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Prepare raw DWI data for the pipeline (format conditioning with MRtrix3)."
    )

    # Required arguments
    parser.add_argument(
        "--data_name",
        required=True,
        choices=["HCP"],
        help="Name of the dataset to use"
    )

    # One of two required arguments must be provided
    sub_opts = parser.add_mutually_exclusive_group(required=True)
    sub_opts.add_argument(
        "--sub_ids",
        nargs="+",
        help="List of subject IDs"
    )
    sub_opts.add_argument(
        "--sub_ids_file",
        help="File containing list of subject IDs, one per line"
    )

    parser.add_argument(
        "--resolution",
        required=True,
        choices=["low", "high"],
        help="Resolution of the data"
    )

    # Optional arguments
    parser.add_argument(
        "--config",
        help="Path to the configuration file"
    )

    parser.add_argument(
        "--personal_workspace",
        help="Personal workspace directory"
    )

    parser.add_argument(
        "--no_raw_backup",
        action="store_false",
        dest="save_raw_data",
        help="Do not save a copy of the raw data"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )

    parser.add_argument(
        '--n_threads',
        type=int,
        default=os.cpu_count() or 4,
        help="Number of threads to use"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory"
    )

    # Parse CLI arguments
    args = parser.parse_args()

    # Get subject IDs
    sub_ids = args.sub_ids or []
    if args.sub_ids_file is not None:
        with open(args.sub_ids_file, "r", encoding="utf-8") as f:
            sub_ids = [line.strip() for line in f if line.strip()]
    sub_ids = list(dict.fromkeys(sub_ids))  # remove duplicates

    # Prepare each subject
    preparer = DWIPreparer(args)
    for sub_id in sub_ids:
        preparer.init_subject(sub_id)
        preparer.run_pipeline(sub_id)


if __name__ == "__main__":
    main()
