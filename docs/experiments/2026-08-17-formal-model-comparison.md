# Formal model comparison for dissertation reporting

Updated: 2026-08-17
Status: verified comparison draft

## Comparable formal 10-fold evidence

The locked from-scratch CNN and AudioSet-pretrained EfficientAT MN20 used the same
UrbanSound8K ten-fold test coverage and cyclic validation-fold mapping. Each test fold was
excluded from training and checkpoint selection, evaluated once, and never used to tune the
method.

| Model | Pretraining | Macro F1 mean ± std | Accuracy mean ± std | Status |
| --- | --- | ---: | ---: | --- |
| From-scratch spectrogram CNN | None | `0.79041 ± 0.04755` | `0.77423 ± 0.05431` | Formal 10-fold |
| EfficientAT MN20 CNN | AudioSet | **`0.87686 ± 0.04048`** | **`0.86883 ± 0.04263`** | Formal 10-fold |

Under the matched formal protocol, AudioSet pretraining improved Macro F1 by `0.08645`
and Accuracy by `0.09460`. It also reduced the between-fold Macro F1 standard deviation by
`0.00707`. These values support transfer learning as the strongest tested formal approach,
but they do not isolate pretraining as the only causal factor because the pretrained system
also uses an EfficientAT backbone and its official waveform/log-Mel frontend.

## Evidence that is not directly comparable

The from-scratch Spectrogram Transformer has a verified fold-10 Accuracy of `0.6547` and
Macro F1 of `0.6644`. It was trained without AudioSet pretraining and performed substantially
below both CNN approaches on that fold. However, it does not yet have a matched formal
10-fold mean and standard deviation, so its single-fold score must not be placed in the same
statistical column as the two formal CNN results.

After the MN20 formal experiment was locked, a six-model MN20+MN40 cross-scale ensemble
reached development validation Macro F1 `0.90104 ± 0.00920` on folds 1, 4 and 7. Fold 10
remained sealed and no new formal test was run. This is a useful post-formal exploratory result,
not a replacement for the locked MN20 formal score `0.87686 ± 0.04048`.

## Interpretation

The from-scratch CNN is a valid baseline but exhibits material fold sensitivity. Its Macro F1
ranged from `0.70436` to `0.85835`, with the earlier fold-10-only result lying near the top of
that range. The aggregate confusion analysis identifies air conditioner, engine idling and
jackhammer as the principal weakness. These classes share persistent low-frequency machinery
content or repetitive impacts, which is consistent with the observed cross-class error counts.

The pretrained MN20 model improved every aggregate class F1 relative to the from-scratch CNN
except that the comparison should be described as system-level rather than a controlled
backbone-only ablation. Its formal gain is large enough to justify presenting AudioSet transfer
learning as the main performance result. The from-scratch CNN remains important because it
shows what can be achieved using only UrbanSound8K supervision, while the Transformer provides
an architecture comparison and illustrates the sample-efficiency limitation of training a
global-attention model from scratch on a small dataset.

## Reporting rules

1. Report formal results as mean ± population standard deviation across ten folds.
2. Do not select or retune any method using the completed formal test results.
3. Label the Transformer score as fold-10-only until a matched multi-fold experiment exists.
4. Label `0.90104 ± 0.00920` as development-only post-formal exploration.
5. Disclose AudioSet pretraining, the EfficientAT checkpoint, and the different official frontend.

## Evidence sources

- From-scratch CNN: `docs/experiments/2026-08-17-cnn-formal-10fold.md`, commit `aef4a4f`.
- EfficientAT MN20: `docs/experiments/pretrained-cnn-recommended-study.md`, commit `f50c98d`.
- Cross-scale exploration: `docs/experiments/pretrained-cnn-bold-breakthrough.md`, commit `d0734b3`.
- Transformer: `results/transformer_baseline_fold10/` and `docs/progress_tracker.md`.
