# EfficientAT MN20 Post-formal Exploration

Status: implementation and development validation pending.

## Research boundary

The locked MN20 formal 10-fold result has already been observed. This follow-up is therefore explicitly post-formal and exploratory. It must not read, rerun, compare, or select against any formal test-fold result. All decisions use only the mean validation Macro F1 across development folds 1, 4, and 7; fold 10 remains sealed in every configuration and runner.

## Fixed control

- Backbone: AudioSet-pretrained EfficientAT MN20.
- Stage: partial fine-tuning of the last two convolution blocks.
- Seeds: 42, 123, and 2026.
- Training: eight epochs, encoder LR `2e-5`, head LR `3e-4`, AdamW, weight decay `1e-4`.
- Regularisation: Mixup alpha `0.15`, probability `0.5`; class weighting and class-aware sampling power `0.5`.
- Selection metric: mean development validation Macro F1.
- Test evaluation: forbidden.

Both loss families reuse the exact same per-seed, per-fold linear-probe checkpoints so that the fine-tuning loss is the only changed training variable.

## Controlled comparisons

1. Three-seed probability ensemble using each seed's validation-selected best checkpoint.
2. Validation top-three checkpoint weight average per seed, restricted to epochs 5--8, followed by single-seed and three-seed probability evaluation.
3. Loss-only replacement of weighted cross-entropy by class-balanced focal loss with gamma `1.5`. Every other model and training parameter remains fixed.

The runner writes a single comparison table containing single-seed best, single-seed checkpoint average, three-seed probability ensemble, and the three-seed ensemble of checkpoint-averaged models for both loss families.

## Command

```bash
python scripts/run_pretrained_cnn_postformal_study.py \
  --backup-root /content/drive/MyDrive/urbansound8k_data/experiment_artifacts/pretrained_cnn_postformal
```

## Results

Pending Colab development runs. No test-fold metric will be added to this study.
