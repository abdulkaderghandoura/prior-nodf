import os
import sys
import argparse
from typing import Tuple

import nibabel as nib
import numpy as np
from dipy.data import get_sphere, HemiSphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines, length
from dipy.tracking import utils as track_utils
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utility import get_mask, load_nifti, Timer, set_default_paths


def compute_tractography(
    fodfs_path: str,
    mask_path: str,
    fa_img_path: str,
    suffix: str = '',
    ckpt_name: str = None,
    output_dir: str = None,
    roi: Tuple[int, int, int, int, int, int] = None,
) -> None:
    """Deterministic tractography on the fODFs; saves the streamlines as a .tck file."""
    # Load mask and affine
    mask, affine = get_mask(mask_path, roi=roi)[:2]

    # Load fODF SH coefficients
    fodf_sh = load_nifti(fodfs_path)[0].astype(np.float32)
    fodf_sh[~mask] = 0

    # Load FA image as reference
    fa_img = nib.load(fa_img_path)
    FA = fa_img.get_fdata(dtype=np.float32)

    # Configs for tractography
    sh_basis = "tournier07"
    theta = 30
    fa_thresh = 0.15
    step_size = 0.5
    min_length = 10
    max_length = 500

    tracking_sphere = HemiSphere.from_sphere(get_sphere("repulsion724"))
    dgklass = DeterministicMaximumDirectionGetter

    print("Creating direction getter for " + sh_basis)
    dg = dgklass.from_shcoeff(
        fodf_sh,
        max_angle=theta,
        sphere=tracking_sphere,
        basis_type=sh_basis,
    )

    threshold_criterion = ThresholdStoppingCriterion(FA, fa_thresh)

    seed_mask = FA >= 0.2
    seeds = track_utils.seeds_from_mask(seed_mask, affine, density=1)

    max_steps = int(max_length / step_size) + 1

    print("==> Performing tractography...")

    with Timer():
        # Generate streamlines using LocalTracking
        streamline_generator = LocalTracking(
            dg,
            threshold_criterion,
            seeds,
            affine,
            step_size=step_size,
            maxlen=max_steps,
            return_all=True,
            random_seed=0,
        )

        # Keep streamlines between min_length and max_length (mm)
        print("Generating and filtering streamlines...")
        streamlines_filtered = Streamlines(
            s for s in streamline_generator if min_length <= length(s) <= max_length
        )

        print(f"Generated {len(streamlines_filtered)} streamlines")

        # Space.RASMM means streamlines are in world coordinates (mm)
        sft = StatefulTractogram(streamlines_filtered, fa_img, Space.RASMM)

        # Save tractogram
        os.makedirs(output_dir, exist_ok=True)
        fname_ext = f"{'_' + ckpt_name if ckpt_name else ''}{suffix if suffix else ''}.tck"
        out_tractogram = os.path.join(output_dir, f"tractography_csd_bv3_FB_ode{fname_ext}")
        save_tractogram(sft, out_tractogram, bbox_valid_check=False)

    print(f"✔  Tractogram saved to: {out_tractogram}", "\n")
    print(f"Number of streamlines: {len(streamlines_filtered)}")


def main():
    parser = argparse.ArgumentParser(description="Run deterministic tractography.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument('--fodfs_path', type=str, help='Path to FODF SH image (.nii.gz)')
    parser.add_argument('--mask_path', type=str, help='Path to mask image (.nii.gz)')
    parser.add_argument('--fa_img_path', type=str, help='Path to FA image')
    parser.add_argument('--suffix', type=str, help='Suffix to include in filename')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory')

    # Use MRtrix (or a similar tool) to determine the region of interest visually
    parser.add_argument("--roi", nargs=6, metavar=('x1', 'x2', 'y1', 'y2', 'z1', 'z2'),
                        type=int, help="Region of interest to compute tractography for")

    args = parser.parse_args()

    # Set suffix
    suffix = f"_{args.suffix}" if args.suffix else ""

    # Set default paths
    defaults = {
        "fodfs_path": os.path.join(args.data_dir, "gt_calcs", f"fodfs_tournier07{suffix}.nii.gz"),
        "mask_path": os.path.join(args.data_dir, "mask.nii.gz"),
        "fa_img_path": os.path.join(args.data_dir, "gt_calcs", f"fa{suffix}.nii.gz"),
        "output_dir": os.path.join(args.data_dir, "gt_calcs"),
    }
    set_default_paths(args, defaults)

    # Compute tractography
    compute_tractography(
        fodfs_path=args.fodfs_path,
        mask_path=args.mask_path,
        fa_img_path=args.fa_img_path,
        suffix=suffix,
        output_dir=args.output_dir,
        roi=args.roi,
    )


if __name__ == '__main__':
    main()
