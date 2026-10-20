# Prior-Guided Implicit Neural Representations for Single-Subject Diffusion MRI Super-Resolution

Abdulkader Ghandoura, Marsil Zakour, William Consagra, Yogesh Rathi

British Machine Vision Conference (BMVC) 2026 · [arXiv:2609.00981](https://arxiv.org/abs/2609.00981) · [Project page](https://www.abdulkaderghandoura.com/research/msc-thesis/)

Clinical diffusion MRI is often acquired with thick slices, which loses fine anatomical detail. This
repository implements a framework that recovers it: an implicit neural representation (INR) pre-trained
on a high-resolution template acts as an anatomical prior and is adapted to each subject through
registration and fine-tuning. For through-plane super-resolution from 5 mm to 1.25 mm, it reconstructs
at higher quality than two recent INR-based methods, NODF and NODF-HashEnc, while training substantially
faster.

## Repository structure

```text
.
├── configs
│   └── datasets.yaml               # Data sources for prepare_dwi.py
├── docs
│   ├── reproducing.md              # Commands for every stage of the experiments
│   └── notes.md                    # Implementation notes
├── src
│   ├── data
│   │   ├── data_module.py          # Coordinates, signal and caches; template mapping for fine-tuning
│   │   └── dataset.py              # Dataset of observation points
│   ├── models
│   │   ├── nodf.py                 # Lightning module: data fidelity, TV regularization, optimizer
│   │   ├── inr.py                  # INR backbone
│   │   ├── siren.py                # Sine layers
│   │   ├── wire.py                 # Gabor wavelet layers
│   │   ├── relu.py                 # ReLU layers
│   │   └── hash_embeddings.py      # Multi-resolution hash encoding (NODF-HashEnc)
│   ├── utils
│   │   ├── utility.py              # Arguments, data loading and shared helpers
│   │   ├── transform.py            # Maps points through the template registration
│   │   ├── custom_callbacks.py     # Epoch and wall-clock checkpoints
│   │   ├── compute_gt.py           # Ground truth: SH coefficients, GFA, DTI, fODFs, tractogram
│   │   ├── shls.py                 # ODF spherical-harmonic coefficients
│   │   ├── gfa.py                  # Generalized fractional anisotropy
│   │   ├── dti.py                  # Diffusion tensor metrics
│   │   ├── fodfs.py                # Fiber ODFs
│   │   ├── tractography.py         # Deterministic tractography
│   │   ├── downsample_throughplane.py  # Thick-slice input by slice averaging
│   │   ├── register.py             # Template registration with uniGradICON
│   │   ├── metrics.py              # Result tables and figure
│   │   └── significance.py         # Paired significance tests
│   ├── prepare_dwi.py              # Data preparation with MRtrix3
│   ├── train.py                    # Training and fine-tuning
│   ├── predict.py                  # ODF coefficients from a checkpoint
│   ├── reconstruct.py              # Diffusion signal from the predicted ODFs
│   └── evaluate.py                 # Metrics, fODFs and tractography
├── LICENSE
├── README.md
├── requirements.txt
└── THIRD_PARTY_LICENSES.md
```

Running the pipeline adds `data/` (prepared data, ground truth and caches), `output/experiments/`
(checkpoints, predictions, reconstructions, evaluations), `logs/` (TensorBoard) and
`assets/registration/` (uniGradICON weights).

## Installation

Tested on Linux with NVIDIA GPUs; on Windows, use WSL2.

```bash
mamba create -n prior-nodf python=3.8.16
mamba activate prior-nodf
pip install -r requirements.txt
```

Two external tools are needed:

- [MRtrix3](https://www.mrtrix.org/) for data preparation.
- [uniGradICON](https://github.com/uncbiag/uniGradICON) for registration, in its own environment
  (Python 3.9 or newer): `pip install unigradicon`.

## Data

The experiments use diffusion MRI from the [Human Connectome Project](https://www.humanconnectome.org/)
Young Adult study (1.25 mm isotropic, three shells of 90 directions, processed with the HCP minimal
preprocessing pipelines), available under the HCP data use terms. Set `source_dir` in
`configs/datasets.yaml`, then prepare each subject:

```bash
python src/prepare_dwi.py --data_name HCP --sub_ids 673455 957974 --resolution high \
    --personal_workspace <directory containing prior-nodf>
```

## Usage

For each subject:

1. Prepare the data and compute the ground truth (`prepare_dwi.py`, `compute_gt.py`).
2. Pre-train the INR on the high-resolution template (`train.py`).
3. Create the thick-slice input by averaging every four axial slices (`downsample_throughplane.py`).
4. Fit a short INR to the low-resolution subject and register the template to it (`train.py`, `register.py`).
5. Fine-tune the template INR on the subject through the registration (`train.py --fine_tune`).
6. Predict, reconstruct and evaluate (`predict.py`, `reconstruct.py`, `evaluate.py`).
7. Aggregate the results over subjects (`metrics.py`, `significance.py`).

The full commands, including both baselines, are in [docs/reproducing.md](docs/reproducing.md);
implementation notes are in [docs/notes.md](docs/notes.md).

## Acknowledgements

This code builds on [NODF](https://github.com/Will-Consagra/NODF) and
[NODF-HashEnc](https://github.com/MunzerDw/nodf-hashenc), used with the permission of their authors.
Components adapted from WIRE, SIREN, NeSVoR and HashNeRF-pytorch are under the MIT License; see
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). The work was carried out at the Psychiatry
Neuroimaging Laboratory, Brigham and Women's Hospital, Harvard Medical School, and formed the first
author's M.Sc. thesis at the Technical University of Munich.

## License

BSD 3-Clause; see [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{Ghandoura_2026_BMVC,
  author    = {Abdulkader Ghandoura and Marsil Zakour and William Consagra and Yogesh Rathi},
  title     = {Prior-Guided Implicit Neural Representations for Single-Subject Diffusion MRI Super-Resolution},
  booktitle = {37th British Machine Vision Conference 2026, {BMVC} 2026, Lancaster, UK, November 23-26, 2026},
  publisher = {BMVA},
  year      = {2026}
}
```
