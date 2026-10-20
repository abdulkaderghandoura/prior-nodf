# Notes

Implementation details that affect how the code is run, with room for improvement.

- **Caches.** Every experiment passed `--preprocess`, which recomputes the cached coordinates in a
  subject's `train_calcs/` and `finetune_calcs/` folders. Without it, existing cache files are reused
  even if they were made with other settings. Keying the caches by their inputs would make the flag
  unnecessary.
- **One run per subject at a time.** Runs on the same subject share these folders, so runs started in
  parallel overwrite each other's cache files. Run them one after another, or give each run its own
  copy of the subject's folder; per-run caches would lift this restriction.
- **Single GPU.** Training uses `devices="auto"`, which starts multi-GPU training when several GPUs
  are visible and changes the effective batch size, so the commands set `CUDA_VISIBLE_DEVICES=0`.
- **Evaluation time.** Besides the reported metrics, `evaluate.py` computes fODFs, fODF peak-count
  metrics and a tractogram, which take much longer. The reported metrics are written to
  `cv_metrics_*.json` before those steps start.
