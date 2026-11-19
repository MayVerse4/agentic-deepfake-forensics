"""
Undersampling balanced sampler -- replaces WeightedRandomSampler.

WeightedRandomSampler(replacement=True) OVERSAMPLES the minority class
(720 real videos, drawn ~5x each per epoch on average to match 3600
fakes) -- with replacement, some real videos get picked far more than
5x in a given epoch, others barely at all, purely by chance. This is a
known source of training instability with heavily imbalanced data
(5:1 here), and critically: it doesn't show up in small-scale tests,
because a 100-per-class diagnostic subset is already balanced 1:1 with
no oversampling happening at all -- which is exactly why every
small-scale check looked clean while every full-scale run collapsed.

This sampler UNDERSAMPLES instead: every epoch, uses ALL real videos
plus a FRESH random subset of fakes of the same size (no replacement,
no repeats within an epoch). Less data per epoch, but stable, honestly
1:1 balanced signal. Since a different random fake subset is drawn each
epoch, the full fake dataset still gets seen over many epochs.
"""

import random
from torch.utils.data import Sampler


class BalancedUndersampleSampler(Sampler):
    def __init__(self, labels, seed=42):
        self.real_idx = [i for i, l in enumerate(labels) if l == 0.0]
        self.fake_idx = [i for i, l in enumerate(labels) if l == 1.0]

        if len(self.real_idx) == 0 or len(self.fake_idx) == 0:
            raise RuntimeError("Both classes must have at least one sample.")

        # Balance to whichever class is smaller -- normally real (minority)
        self.n = min(len(self.real_idx), len(self.fake_idx))
        self._epoch = 0
        self._base_seed = seed

    def __iter__(self):
        rng = random.Random(self._base_seed + self._epoch)
        self._epoch += 1

        real_sample = self.real_idx if len(self.real_idx) == self.n else rng.sample(self.real_idx, self.n)
        fake_sample = self.fake_idx if len(self.fake_idx) == self.n else rng.sample(self.fake_idx, self.n)

        combined = list(real_sample) + list(fake_sample)
        rng.shuffle(combined)
        return iter(combined)

    def __len__(self):
        return self.n * 2
