# EfficientAT MN20 Post-formal Exploration

Status: development validation completed on 2026-08-17; no test fold was evaluated.

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

All 27 training runs completed on a Colab A100 between `2026-08-17T11:56:00Z` and `2026-08-17T12:16:27Z`. This comprises nine shared linear-probe initialisations, nine weighted cross-entropy fine-tuning runs, and nine focal-loss fine-tuning runs. The summed run time recorded in the manifests was 1,092.6 seconds. All 27 manifests report `test_evaluated=false`; the aggregate summary also reports `formal_test_results_used_for_selection=false`.

| Loss and aggregation method | Validation Macro F1 mean | F1 std | Validation Accuracy mean | Accuracy std |
| --- | ---: | ---: | ---: | ---: |
| Weighted CE, seed 42 best checkpoint | 0.89069 | 0.01165 | 0.88742 | 0.01062 |
| Weighted CE, seed 42 checkpoint average | 0.88438 | 0.01004 | 0.88230 | 0.01051 |
| Weighted CE, three-seed probability ensemble | 0.89267 | 0.01052 | 0.88900 | 0.00933 |
| Weighted CE, ensemble of checkpoint averages | 0.88618 | 0.01033 | 0.88311 | 0.00909 |
| Focal loss, seed 42 best checkpoint | 0.89125 | 0.00906 | 0.88776 | 0.01154 |
| Focal loss, seed 42 checkpoint average | 0.88324 | 0.00753 | 0.88044 | 0.01155 |
| **Focal loss, three-seed probability ensemble** | **0.89395** | **0.00932** | **0.89004** | **0.01062** |
| Focal loss, ensemble of checkpoint averages | 0.88312 | 0.00841 | 0.87964 | 0.01183 |

The winning focal-loss probability ensemble achieved per-fold Macro F1 values of `0.90014`, `0.90093`, and `0.88078` on folds 1, 4, and 7, respectively. The corresponding weighted-CE ensemble values were `0.90182`, `0.89826`, and `0.87793`.

## Interpretation

The three-seed probability ensemble provided a small, consistent gain for both losses. Under weighted CE it improved Macro F1 from `0.89069` to `0.89267` (`+0.00198`) and reduced the standard deviation from `0.01165` to `0.01052`. Replacing weighted CE with class-balanced focal loss produced only a small single-seed gain (`+0.00056`), but combining focal loss with the three-seed probability ensemble reached `0.89395 ± 0.00932`. This is `+0.00327` above the locked development control and has slightly lower fold variation.

Checkpoint weight averaging was not effective. It reduced mean Macro F1 by approximately `0.0063`--`0.0081` relative to the corresponding validation-selected best checkpoints. The likely explanation is that the short fine-tuning trajectory crosses distinct local states; arithmetic parameter averaging across validation-ranked epochs does not guarantee a better point in function space.

The improvement is modest and remains post-formal exploratory evidence because the formal 10-fold result had already been observed before this study was designed. It cannot replace the locked formal result of `0.87686 ± 0.04048`, and no new test metric is reported. A future confirmatory study would need a newly specified protocol or an untouched external test set. The complete artifacts are backed up under `pretrained_cnn_postformal/` in Google Drive; generated checkpoints, results, figures, and waveform caches are not committed to Git.
