# EfficientAT attribution

The files under `src/vendor/efficientat/` are adapted from EfficientAT at
upstream commit `a425fdce92572e602a1d5634799bd9f1f2efa806`:

https://github.com/fschmid56/EfficientAT

EfficientAT is Copyright (c) 2022 Florian Schmid and distributed under the
MIT License reproduced in `third_party/efficientat/LICENSE`.

Local changes are limited to package-relative imports, removal of constructor
printing, and use of the current complex-valued `torch.stft` API. The feature
extraction calculation and MobileNet architecture are otherwise preserved.
