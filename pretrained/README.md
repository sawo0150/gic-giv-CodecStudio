# Pretrained Instant-GI Model

Place `epoch_best_ks_3.pth` in this directory.

Run:

```bash
python scripts/setup_models.py
```

The setup script mirrors the checkpoint into `Instant-GI/checkpoints/` when needed.

The codec also includes a NumPy fallback backend so `.gic` and `.giv` demos can run in environments where Torch or the custom Instant-GI CUDA extensions are unavailable.
