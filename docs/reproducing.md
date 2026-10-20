# Reproducing the experiments

The commands below run the full pipeline for one test subject, HCP `673455`, with the template
`957974`, from the repository root. The paper reports the mean over ten test subjects: `214221`,
`163432`, `389357`, `120717`, `744553`, `393550`, `882161`, `540436`, `465852` and `673455`. For
another subject, replace `673455`.

Each training run writes to `output/experiments/<experiment>/<model>/version_<N>/`. The commands
assume the first run of each experiment (`version_0`); later runs increment `N`, which `train.py`
prints when it starts. `register.py` runs in the uniGradICON environment, everything else in the main
environment.

## 1. Data and ground truth

```bash
python src/prepare_dwi.py --data_name HCP --sub_ids 673455 957974 --resolution high \
    --personal_workspace <directory containing prior-nodf>

CUDA_VISIBLE_DEVICES=0 python src/utils/compute_gt.py --data_dir data/HCP/processed/sub-673455/HighRes/ --bval 1000 --tolerance 50
CUDA_VISIBLE_DEVICES=0 python src/utils/compute_gt.py --data_dir data/HCP/processed/sub-957974/HighRes/ --bval 1000 --tolerance 50
```

## 2. Template pre-training

```bash
CUDA_VISIBLE_DEVICES=0 python src/train.py --bval 1000 --tolerance 50 --batch_size 128 --num_epochs 3000 \
    --num_workers 4 --pin_memory --data_dir data/HCP/processed/sub-957974/HighRes --use_baseline \
    --r 1024 --depth 10 --learning_rate 1e-6 --omega0 60 --experiment_name template --preprocess
```

Fine-tuning starts from the checkpoint after 2,500 epochs, `epoch_2499-step_15465000.ckpt`
(checkpoints are saved every 500 epochs).

## 3. Thick-slice input

```bash
python src/utils/downsample_throughplane.py --data_dir data/HCP/processed/sub-673455/HighRes/ \
    --output_dir data/HCP/processed/sub-673455/LowSpatialFactor4HighAngular/
```

## 4. Registration

A short fit to the low-resolution subject provides the upsampled GFA that the template is registered to.

```bash
CUDA_VISIBLE_DEVICES=0 python src/train.py --bval 1000 --tolerance 50 --batch_size 128 --num_epochs 25 \
    --num_workers 4 --pin_memory --data_dir data/HCP/processed/sub-673455/LowSpatialFactor4HighAngular/ \
    --use_baseline --r 1024 --depth 10 --learning_rate 1e-6 --preprocess --omega0 60 --experiment_name upsample

for step in predict reconstruct evaluate; do
    CUDA_VISIBLE_DEVICES=0 python src/$step.py \
        --ckpt_path output/experiments/upsample/nodf/version_0/training/checkpoints/epoch_20_snapshot.ckpt \
        --preprocess --continuous --reslice_like data/HCP/processed/sub-673455/HighRes/b0.nii.gz \
        --ref_data_dir data/HCP/processed/sub-673455/HighRes/
done

# in the uniGradICON environment
CUDA_VISIBLE_DEVICES=0 python src/utils/register.py \
    --fixed_path output/experiments/upsample/nodf/version_0/evaluation/gfa_epoch_20_snapshot_resliced_1p25x1p25x1p25_like_HCP_sub-673455_HighRes.nii.gz \
    --moving_path data/HCP/processed/sub-957974/HighRes/gt_calcs/gfa.nii.gz --io_iterations 50 --suffix 20_epochs
```

## 5. Proposed method

```bash
CUDA_VISIBLE_DEVICES=0 python src/train.py --bval 1000 --tolerance 50 --batch_size 128 --num_epochs 100 \
    --num_workers 4 --pin_memory --experiment_name proposed \
    --data_dir data/HCP/processed/sub-673455/LowSpatialFactor4HighAngular/ \
    --transform output/experiments/upsample/nodf/version_0/unigradicon/registered_gfa_template_HCP_sub-957974_HighRes_gt_calcs_model_unigradicon_io_50_sim_lncc_20_epochs.hdf5 \
    --ckpt_path output/experiments/template/nodf/version_0/training/checkpoints/epoch_2499-step_15465000.ckpt \
    --fine_tune --moving data/HCP/processed/sub-957974/HighRes/b0.nii.gz \
    --fixed data/HCP/processed/sub-673455/LowSpatialFactor4HighAngular/b0.nii.gz \
    --use_baseline --r 1024 --depth 10 --learning_rate 1e-6 --omega0 60 \
    --downsampling_subgrid_size 4 4 16 --tv_directions xyz --lambda_tv 0.002 --offset_tv 0.01 --preprocess

for step in predict reconstruct evaluate; do
    CUDA_VISIBLE_DEVICES=0 python src/$step.py \
        --ckpt_path output/experiments/proposed/nodf/version_0/training/checkpoints/<CHECKPOINT>.ckpt \
        --preprocess --continuous --reslice_like data/HCP/processed/sub-673455/HighRes/b0.nii.gz \
        --ref_data_dir data/HCP/processed/sub-673455/HighRes/
done
```

## 6. Baselines

NODF:

```bash
CUDA_VISIBLE_DEVICES=0 python src/train.py --bval 1000 --tolerance 50 --num_epochs 3000 --num_workers 4 \
    --pin_memory --experiment_name baseline --data_dir data/HCP/processed/sub-673455/LowSpatialFactor4HighAngular/ \
    --use_baseline --r 1024 --depth 10 --learning_rate 1e-6 --preprocess

for step in predict reconstruct evaluate; do
    CUDA_VISIBLE_DEVICES=0 python src/$step.py \
        --ckpt_path output/experiments/baseline/nodf/version_0/training/checkpoints/<CHECKPOINT>.ckpt \
        --preprocess --continuous --reslice_like data/HCP/processed/sub-673455/HighRes/b0.nii.gz \
        --ref_data_dir data/HCP/processed/sub-673455/HighRes/
done
```

NODF-HashEnc:

```bash
CUDA_VISIBLE_DEVICES=0 python src/train.py --bval 1000 --tolerance 50 --num_epochs 3000 --num_workers 4 \
    --pin_memory --experiment_name related_work --data_dir data/HCP/processed/sub-673455/LowSpatialFactor4HighAngular/ \
    --model nodf-hashenc --preprocess

for step in predict reconstruct evaluate; do
    CUDA_VISIBLE_DEVICES=0 python src/$step.py \
        --ckpt_path output/experiments/related_work/nodf-hashenc/version_0/training/checkpoints/<CHECKPOINT>.ckpt \
        --preprocess --continuous --reslice_like data/HCP/processed/sub-673455/HighRes/b0.nii.gz \
        --ref_data_dir data/HCP/processed/sub-673455/HighRes/
done
```

## Checkpoints

Training saves wall-clock snapshots, `time_<MMMMM>min_epoch<E>.ckpt` (after 2, 5, 10, 30, 60, 120 and
180 minutes), and epoch snapshots, `epoch_<E>_snapshot.ckpt`. Set `<CHECKPOINT>` to one of them. The
paper's time budgets were measured on an NVIDIA GeForce GTX 1080 Ti, so the epoch a snapshot reaches
depends on the GPU, and on the subject. To stop at a fixed epoch instead, train with
`--num_epochs <E+1> --snapshot_epochs <E> --snapshot_minutes 100000` and use `epoch_<E>_snapshot`.

## 7. Tables and significance

```bash
python src/utils/metrics.py --input <config.json>
python src/utils/significance.py output/artifacts/metrics/cv_metrics_resolved.json
```

The configuration lists, for each method and time budget, the test subjects' `cv_metrics_*.json`
files written by `evaluate.py` (wildcards allowed):

```json
{
  "HCP": {
    "baseline": {"5_min": ["<subject 1>.json", "..."], "10_min": ["..."], "30_min": ["..."]},
    "hashenc":  {"5_min": ["..."], "10_min": ["..."], "30_min": ["..."]},
    "proposed": {"5_min": ["..."], "10_min": ["..."], "30_min": ["..."]}
  }
}
```

`metrics.py` writes the LaTeX tables, the figure of the median metrics and
`cv_metrics_resolved.json`, which `significance.py` reads.
