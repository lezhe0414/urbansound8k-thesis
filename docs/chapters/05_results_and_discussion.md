# 5. Results and Discussion

## 5.1 Evaluation protocol

Macro F1 was the primary metric because UrbanSound8K classes are not equally represented and
the study aims to avoid improvements that apply only to frequent classes. Accuracy, macro
precision and macro recall were reported as supporting measures. The final from-scratch CNN
configuration was selected before formal evaluation using validation Macro F1. It was then held
fixed for ten-fold cross-validation: every official fold served as the test set exactly once, the
next fold served as validation, and the remaining eight folds were used for training. A test fold
was never used for checkpoint selection or hyperparameter tuning.

## 5.2 Formal results

| Model | Protocol | Macro F1 | Accuracy |
| --- | --- | ---: | ---: |
| From-scratch spectrogram CNN | Formal 10-fold | `0.79041 ± 0.04755` | `0.77423 ± 0.05431` |
| EfficientAT MN20, AudioSet pretrained | Formal 10-fold | **`0.87686 ± 0.04048`** | **`0.86883 ± 0.04263`** |
| From-scratch Spectrogram Transformer | Fold 10 only | `0.6644` | `0.6547` |

The pretrained CNN exceeded the from-scratch CNN by `0.08645` Macro F1 and `0.09460`
Accuracy under the matched formal ten-fold protocol. Its lower fold standard deviation also
indicates more stable generalisation across the recording-condition partitions. The Transformer
result is included only as architecture evidence; it lacks a matched ten-fold estimate and must
not be interpreted as a directly comparable mean.

## 5.3 From-scratch CNN fold variability

The CNN Macro F1 ranged from `0.70436` on fold 3 to `0.85835` on fold 10. The previously
reported single fold-10 score of `0.8536` was therefore close to the upper end of the formal fold
distribution and exceeded the ten-fold mean by approximately `0.0632`. This demonstrates why
the earlier score was insufficient as the headline result. The aggregate prediction set produced
Macro F1 `0.79091` and Accuracy `0.77279`; the small difference from the fold means results from
unequal fold sizes.

The weakest aggregate class F1 values were air conditioner (`0.61010`), jackhammer
(`0.64812`) and engine idling (`0.68759`). The strongest were gun shot (`0.93782`) and car
horn (`0.90196`). The most frequent directional errors were air conditioner classified as engine
idling (`160` clips), jackhammer as air conditioner (`159`), engine idling as air conditioner
(`132`) and engine idling as jackhammer (`131`). The remaining errors are therefore not random:
the principal limitation is separating acoustically related machinery classes.

## 5.4 Training behaviour

Validation-selected checkpoints occurred at epochs 8--10, with a mean best epoch of `9.1`.
At those checkpoints, training Macro F1 averaged `0.82670` and validation Macro F1 averaged
`0.79833`. The apparent gap of `0.02837` is modest, but it cannot be treated as a conventional
clean train-validation gap because training batches contain Mixup, SpecAugment and class-aware
resampling. Several folds had validation F1 above the augmented training F1. The evidence does
not support severe universal overfitting; fold-dependent generalisation is the larger issue.

## 5.5 Transfer learning and exploratory extension

The formal EfficientAT result supports the use of AudioSet transfer learning for this dataset.
The gain should be described as a comparison between complete systems rather than proof that
pretraining alone caused the entire difference, since EfficientAT also changes the backbone and
uses its official waveform-based log-Mel frontend. Nevertheless, both formal systems used the
same official UrbanSound8K fold coverage and test-sealing rule, making the observed performance
difference meaningful for the research question.

A later development-only six-model MN20+MN40 cross-scale ensemble reached Macro F1
`0.90104 ± 0.00920` on validation folds 1, 4 and 7. This was conducted after formal MN20
evaluation, did not evaluate fold 10 and did not use ten-fold test data. It is reported as
post-formal exploratory evidence of complementary model scales, not as a new formal result.

## 5.6 Limitations

The main limitations are the small dataset, environmental differences between official folds,
the absence of a matched ten-fold Transformer experiment, and the computational cost of
multi-seed pretrained ensembles. The formal results must not be used for further model selection.
Future work should validate the fixed cross-scale ensemble on a new untouched benchmark or a
separately preregistered evaluation dataset, and should examine class-specific features for the
air-conditioner, engine-idling and jackhammer confusion cluster.
