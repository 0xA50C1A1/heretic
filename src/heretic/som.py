# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from collections import defaultdict

import numpy as np
from minisom import MiniSom


class SOMCalculator:
    """Trains a Self-Organizing Map and extracts the most-used neuron weights."""

    def __init__(
        self,
        som_x: int,
        som_y: int,
        iterations: int,
        lr: float,
        sigma: float,
    ) -> None:
        self.som_x = som_x
        self.som_y = som_y
        self.iterations = iterations
        self.lr = lr
        self.sigma = sigma
        self.som = None
        self._data = None

    def fit(self, data: np.ndarray) -> None:
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got shape {data.shape}")

        self._data = data
        n_features = data.shape[1]

        self.som = MiniSom(
            self.som_x,
            self.som_y,
            n_features,
            sigma=self.sigma,
            learning_rate=self.lr,
            random_seed=0,
            activation_distance="euclidean",
            topology="hexagonal",
        )
        self.som.random_weights_init(data)
        self.som.train_random(data, self.iterations)

    def get_top_k_neuron_weights(self, k: int) -> np.ndarray:
        if self.som is None or self._data is None:
            raise RuntimeError(
                "SOM has not been trained yet. Call `fit()` first.")

        # Count how often each neuron wins across the actual training samples.
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for x in self._data:
            counts[tuple(self.som.winner(x))] += 1

        sorted_neurons = sorted(
            counts.items(), key=lambda item: item[1], reverse=True
        )[:k]

        weights = self.som.get_weights()  # (som_x, som_y, n_features)
        top_k = np.array([weights[i, j] for (i, j), _ in sorted_neurons])
        return top_k
