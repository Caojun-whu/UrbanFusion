#!/usr/bin/env python3
"""
Description: Creates a dataset with two modalities, each represented as a
3-dimensional vector. The first two dimensions contain redundant information
(shared between modalities, useful for geolocalization). The third dimension
contains modality-specific information: during evaluation, it is random noise;
during training, it is a constant value across the entire batch.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset


def _constant_blue_collate(batch):
    """
    Collate a list of sample dictionaries into a batch dictionary of tensors,
    overwriting dimension 2 ("blue") with a constant random value shared across
    all samples for each modality.

    For each modality present in the batch ("mod1", "mod2"), the function:
    1. Stacks the individual sample tensors along a new batch dimension.
    2. Draws a single value from U(0, 1).
    3. Broadcasts this value into the blue dimension (index 2) for all samples.

    Parameters
    ----------
    batch : list
        A list of sample dictionaries, where each key corresponds to a modality
        (e.g., "mod1", "mod2") and maps to a tensor of shape (..., C), with
        C ≥ 3 (at least three dimensions).

    Returns
    -------
    dict
        A dictionary with the same keys as the input batch. Each value is a
        batched tensor with the blue dimension replaced by a constant random
        value per modality.

    Notes
    -----
    - The blue dimension value is drawn independently for each modality.
    - The same constant is applied to all samples within the batch for that
      modality.
    - Intended for use in synthetic data experiments where dimension 2 encodes
      modality-specific information.
    """
    out = {k: torch.stack([d[k] for d in batch], dim=0) for k in batch[0]}

    blue1 = torch.rand(1)
    blue2 = torch.rand(1)

    if "mod1" in out:
        out["mod1"][:, 2] = blue1
    if "mod2" in out:
        out["mod2"][:, 2] = blue2

    return out


def generate_grid(
    lat0: float = 37.0, lon0: float = -122.0, span: float = 0.1, N: int = 51
) -> tuple:
    """
    Generate a grid of latitude and longitude coordinates.

    Parameters
    ----------
    lat0 : float
        Center latitude.
    lon0 : float
        Center longitude.
    span : float
        Span of the grid (in degrees).
    N : int
        Number of points along each dimension.

    Returns
    -------
    tuple
        A tuple containing:
        - coords: (N*N, 2) array of (lat, lon) pairs
        - lat2d: (N, N) array of latitudes
        - lon2d: (N, N) array of longitudes
    """
    lat = np.linspace(lat0 - span / 2, lat0 + span / 2, N)
    lon = np.linspace(lon0 - span / 2, lon0 + span / 2, N)
    lon2d, lat2d = np.meshgrid(lon, lat)
    coord = np.stack([lat2d, lon2d], axis=-1)
    return coord.reshape(-1, 2), lat2d, lon2d


def _norm(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a 1D array to the range [0, 1].

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Normalized array.
    """
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)


def _unique_tile(
    arr2d: np.ndarray,
    *,
    seed: int | None = None,
    morans_I: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Generate a continuous random field in the range [0, 1] with a specified
    level of spatial autocorrelation (Moran's I).

    The spatial structure of the output varies with `morans_I`:

        +1.0  → perfectly positive autocorrelation (uniform field)
         0.0  → spatially random (white noise)
        -1.0  → perfectly negative autocorrelation (checkerboard pattern)

    Intermediate values produce partially correlated fields, with positive
    values yielding smooth blobs and negative values yielding disrupted
    checkerboard patterns.

    Parameters
    ----------
    arr2d : np.ndarray
        Input array whose `shape` determines the spatial dimensions of
        the generated field. Its contents are ignored.
    seed : int or None, default=None
        Random seed for reproducibility.
    morans_I : float, default=0.0
        Desired Moran’s I, clamped to the interval [-1, 1].
    scale : float, default=1.0
        Scalar factor to multiply the generated field.

    Returns
    -------
    np.ndarray
        A float array of shape `arr2d.shape` with values in [0, 1] scaled by
        `scale`.

    Notes
    -----
    - Positive `morans_I` values create spatially smooth structures by
      Gaussian blurring uniform noise.
    - Negative `morans_I` values start from a binary checkerboard pattern and
      progressively randomize it as the value approaches 0.
    - Extreme values ±1.0 yield perfectly uniform or perfectly alternating
      patterns, respectively.
    """
    morans_I = float(np.clip(morans_I, -1.0, 1.0))
    rng = np.random.default_rng(seed)
    shape = arr2d.shape

    # perfect extremes
    if morans_I >= 0.999:  # I ≈ +1 ⇒ constant field
        val = rng.random()
        return np.full(shape, val, dtype=float) * scale

    if morans_I <= -0.999:  # I ≈ –1 ⇒ perfect checkerboard
        even_mask = (np.indices(shape).sum(axis=0) & 1) == 0
        field = np.empty(shape, dtype=float)
        field[even_mask] = rng.uniform(0.0, 0.5, even_mask.sum())
        field[~even_mask] = rng.uniform(0.5, 1.0, (~even_mask).sum())
        return field * scale

    # positive autocorrelation  (0 ≤ I < 1)
    if morans_I >= 0.0:
        noise = rng.random(shape)  # U(0, 1) white noise
        # sigma grows with I – larger sigma ⇒ larger, smoother blobs
        max_sigma = min(shape) / 4.0
        sigma = morans_I * max_sigma + 1e-6
        field = gaussian_filter(noise, sigma=sigma, mode="wrap")
        # rescale to [0, 1]
        field -= field.min()
        field /= field.ptp() + 1e-12
        return field * scale

    # negative autocorrelation  (‑1 < I < 0)
    # Start with a checkerboard of low/high values, then randomise a fraction
    # p_flip so that I → 0⁻ smoothly.
    even_mask = (np.indices(shape).sum(axis=0) & 1) == 0
    field = np.empty(shape, dtype=float)
    field[even_mask] = rng.uniform(0.0, 0.5, even_mask.sum())
    field[~even_mask] = rng.uniform(0.5, 1.0, (~even_mask).sum())

    p_flip = 1.0 - abs(morans_I)
    flip_mask = rng.random(shape) < p_flip
    field[flip_mask] = rng.random(np.count_nonzero(flip_mask))
    return field * scale


def generate_modalities(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    *,
    noise_sigma: float = 0.02,
    seed: int | None = 42,
    morans_I: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two synthetic RGB modalities from latitude and longitude grids.

    The red and green dimensions encode normalized spatial coordinates in
    different linear combinations, providing shared geolocation information
    between modalities. The blue channel is modality-specific and generated
    as a continuous random field with a specified spatial autocorrelation
    (Moran's I). Optional Gaussian noise is added to all dimensions.

    Parameters
    ----------
    lat2d : np.ndarray
        2D array of latitude values.
    lon2d : np.ndarray
        2D array of longitude values.
    noise_sigma : float, default=0.02
        Standard deviation of additive Gaussian noise applied independently
        to each channel in both modalities. Set to 0 for noise-free output.
    seed : int or None, default=42
        Random seed controlling both the autocorrelated blue-channel fields
        and the additive noise. If None, results will be non-deterministic.
    morans_I : float, default=0.0
        Target Moran’s I for the continuous blue dimensions, clamped to the
        interval [-1, 1].

    Returns
    -------
    mod1 : np.ndarray
        First RGB modality, float values in [0, 1], shape `(H, W, 3)`.
    mod2 : np.ndarray
        Second RGB modality, float values in [0, 1], shape `(H, W, 3)`.

    Notes
    -----
    - The red/green dimensions contain redundant geolocation information.
    - The blue dimensions is unique to each modality and generated using
      `_unique_tile`.
    - When `morans_I > 0`, blue dimensions contain smooth spatial blobs;
      when `morans_I < 0`, they contain structured alternating patterns.
    """
    lat_n, lon_n = _norm(lat2d), _norm(lon2d)

    # Mod‑1
    mod1_r, mod1_g = lat_n, lon_n
    mod1_b = _unique_tile(lat2d, seed=seed, morans_I=morans_I)

    # Mod‑2
    mod2_r = 0.5 * (lat_n + lon_n)
    mod2_g = 0.5 * (lat_n - lon_n) + 0.5
    mod2_b = _unique_tile(
        lon2d, seed=None if seed is None else seed + 1, morans_I=morans_I
    )

    mod1 = np.stack([mod1_r, mod1_g, mod1_b], axis=-1)
    mod2 = np.stack([mod2_r, mod2_g, mod2_b], axis=-1)

    # Optional additive Gaussian noise
    if noise_sigma:
        rng = np.random.default_rng(seed)
        mod1 += noise_sigma * rng.standard_normal(mod1.shape)
        mod2 += noise_sigma * rng.standard_normal(mod2.shape)

    return np.clip(mod1, 0, 1), np.clip(mod2, 0, 1)


def split_indices(
    N: int,
    crg_quadrant: int = 0,
    train_fraction: float = 0.8,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split an N×N grid of indices into training, validation-in-region, and
    cross-region generalization (CRG) sets.

    The CRG set consists of all indices in one held-out quadrant of the grid.
    The remaining quadrants are used for in-region training and validation
    according to `train_fraction`.

    Parameters
    ----------
    N : int
        Grid resolution. The total number of elements is `N * N`.
    crg_quadrant : int, default=0
        Which quadrant (0–3) to hold out entirely for CRG testing:
            0 → northwest (top-left)
            1 → northeast (top-right)
            2 → southwest (bottom-left)
            3 → southeast (bottom-right)
    train_fraction : float, default=0.8
        Fraction of in-region samples (non-CRG) assigned to the training set.
        The remainder are assigned to the validation-in-region set.
    seed : int or None, default=42
        Random seed for shuffling in-region samples before splitting. If None,
        results are non-deterministic.

    Returns
    -------
    train_idx : np.ndarray
        1D array of flattened grid indices used for training.
    val_idx : np.ndarray
        1D array of flattened grid indices used for validation within the
        training region.
    crg_idx : np.ndarray
        1D array of flattened grid indices used for CRG testing.

    Notes
    -----
    - The grid is divided into four quadrants of size `(N//2) × (N//2)`.
    - The CRG quadrant is held out completely to test generalization to an
      unseen spatial region.
    - In-region indices are randomly shuffled before splitting into train/val.
    """
    if seed is not None:
        np.random.seed(seed)
    idx_grid = np.arange(N * N).reshape(N, N)
    mid = N // 2
    quads = {
        0: idx_grid[:mid, :mid].ravel(),  # NW
        1: idx_grid[:mid, mid:].ravel(),  # NE
        2: idx_grid[mid:, :mid].ravel(),  # SW
        3: idx_grid[mid:, mid:].ravel(),  # SE
    }
    crg_idx = quads[crg_quadrant]
    in_region = np.concatenate(
        [v for k, v in quads.items() if k != crg_quadrant]
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(in_region)

    split_ir = int(train_fraction * in_region.size)
    train_idx, val_idx = in_region[:split_ir], in_region[split_ir:]

    return train_idx, val_idx, crg_idx


def plot_modalities(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    mod1: np.ndarray,
    mod2: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    crg_idx: np.ndarray,
    save: bool = False,
    prefix: str = "fig",
) -> None:
    """
    Plot two RGB modalities, their individual channels, and spatial sampling
    splits for training, validation, and cross-region generalization (CRG).

    The figure consists of:
    - Row 0: RGB composites for each modality, plus a scatter plot showing
      the spatial distribution of train/val/CRG samples.
    - Row 1: Individual channels of modality 1.
    - Row 2: Individual channels of modality 2.

    Parameters
    ----------
    lat2d : np.ndarray
        2D array of latitude coordinates with shape (H, W).
    lon2d : np.ndarray
        2D array of longitude coordinates with shape (H, W).
    mod1 : np.ndarray
        First RGB modality, shape (H, W, 3), values in [0, 1].
    mod2 : np.ndarray
        Second RGB modality, shape (H, W, 3), values in [0, 1].
    train_idx : np.ndarray
        1D array of flattened grid indices for training samples.
    val_idx : np.ndarray
        1D array of flattened grid indices for validation samples within the
        training region.
    crg_idx : np.ndarray
        1D array of flattened grid indices for cross-region generalization
        samples (held-out quadrant).
    save : bool, default=False
        If True, saves the plot to disk as a PNG file.
    prefix : str, default="fig"
        Filename prefix for saved plots; the suffix "_modalities.png" is
        automatically appended.
    """
    extent = [lon2d.min(), lon2d.max(), lat2d.min(), lat2d.max()]

    # Define fewer ticks (e.g., 5 for each axis)
    xticks = np.linspace(extent[0], extent[1], 3)
    yticks = np.linspace(extent[2], extent[3], 3)

    fig, axes = plt.subplots(3, 3, figsize=(12, 11), sharex=True, sharey=True)

    # Row 0 – composites + sample map
    axes[0, 0].imshow(mod1, origin="lower", extent=extent, aspect="auto")
    axes[0, 0].set_title("Modality 1: Composite (RGB)", fontsize=14)
    axes[0, 1].imshow(mod2, origin="lower", extent=extent, aspect="auto")
    axes[0, 1].set_title("Modality 2: Composite (RGB)", fontsize=14)
    ax = axes[0, 2]
    ax.set_title("Samples", fontsize=14)
    ax.scatter(
        lon2d.ravel()[train_idx],
        lat2d.ravel()[train_idx],
        s=4,
        label="Train",
        alpha=0.6,
    )
    ax.scatter(
        lon2d.ravel()[val_idx],
        lat2d.ravel()[val_idx],
        s=1,
        label="Validation",
        alpha=0.6,
    )
    ax.scatter(
        lon2d.ravel()[crg_idx],
        lat2d.ravel()[crg_idx],
        s=4,
        label="Cross-Regional",
        alpha=0.6,
    )
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Train",
            markerfacecolor="C0",
            markeredgewidth=0,
            markeredgecolor="none",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Validation",
            markerfacecolor="C1",
            markeredgewidth=0,
            markeredgecolor="none",
            markersize=6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Cross-Regional",
            markerfacecolor="C2",
            markeredgewidth=0,
            markeredgecolor="none",
            markersize=6,
        ),
    ]

    ax.legend(handles=legend_elements, fontsize=12, loc="upper right")

    # Row 1 – Modality 1 channels
    titles1 = [
        "Modality 1: Dim 1 (lat)",
        "Modality 1: Dim 2 (lon)",
        "Modality 1: Dim 3 (U₁)",
    ]
    cmap1 = ["Reds", "Greens", "Blues"]
    for j in range(3):
        axes[1, j].imshow(
            mod1[..., j],
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap1[j],
        )
        axes[1, j].set_title(titles1[j], fontsize=14)

    # Row 2 – Modality 2 channels
    titles2 = [
        "Modality 2: Dim 1 ((lat+lon)/2)",
        "Modality 2: Dim 2 ((lat−lon)/2+0.5)",
        "Modality 2: Dim 3 (U₂)",
    ]
    cmap2 = cmap1
    for j in range(3):
        axes[2, j].imshow(
            mod2[..., j],
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap2[j],
        )
        axes[2, j].set_title(titles2[j], fontsize=14)

    # Layout tweaks
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if i == 2:
                ax.set_xlabel("Longitude", fontsize=12)
            if j == 0:
                ax.set_ylabel("Latitude", fontsize=12)
            ax.ticklabel_format(style="plain", useOffset=False)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.label_outer()
            ax.tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout()
    if save:
        plt.savefig(f"{prefix}_modalities.png", dpi=200, bbox_inches="tight")
    plt.show()


def build_data_dict(
    lat0: float = 37.0,
    lon0: float = -122.0,
    span: float = 0.2,
    N: int = 51,
    noise_sigma: float = 0.02,
    crg_quadrant: int = 0,
    train_fraction: float = 0.8,
    seed: int | None = 42,
) -> dict:
    """
    Construct a synthetic geospatial dataset dictionary containing coordinates,
    two RGB modalities, and index splits for training, validation, and
    cross-region generalization (CRG).

    Parameters
    ----------
    lat0 : float, default=37.0
        Center latitude of the generated grid.
    lon0 : float, default=-122.0
        Center longitude of the generated grid.
    span : float, default=0.2
        Total latitude/longitude span of the grid in degrees. The grid extends
        ±`span/2` from the center.
    N : int, default=51
        Grid resolution. The grid contains `N × N` points.
    noise_sigma : float, default=0.02
        Standard deviation of additive Gaussian noise applied to both
        modalities.
    crg_quadrant : int, default=0
        Quadrant index (0–3) to hold out for CRG testing:
            0 → northwest (top-left)
            1 → northeast (top-right)
            2 → southwest (bottom-left)
            3 → southeast (bottom-right)
    train_fraction : float, default=0.8
        Fraction of in-region (non-CRG) samples assigned to the training set.
        The remainder are assigned to validation.
    seed : int or None, default=42
        Random seed controlling both index splitting and noise generation.
        If None, results are non-deterministic.

    Returns
    -------
    dict
        Dictionary containing:
            - `"coord"` : np.ndarray, shape (N*N, 2) — [latitude, longitude]
              pairs.
            - `"lat2d"` : np.ndarray, shape (N, N) — latitude grid.
            - `"lon2d"` : np.ndarray, shape (N, N) — longitude grid.
            - `"mod1"` : np.ndarray, shape (N*N, 3) — modality 1 RGB features.
            - `"mod2"` : np.ndarray, shape (N*N, 3) — modality 2 RGB features.
            - `"train_idx"` : np.ndarray — flattened indices for training
              samples.
            - `"val_idx"` : np.ndarray — flattened indices for validation
              samples.
            - `"crg_idx"` : np.ndarray — flattened indices for CRG testing
              samples.
    """
    coord, lat2d, lon2d = generate_grid(lat0, lon0, span, N)
    mod1, mod2 = generate_modalities(lat2d, lon2d, noise_sigma=noise_sigma)
    train_idx, val_idx, crg_idx = split_indices(
        N, crg_quadrant, train_fraction, seed
    )
    return {
        "coord": coord,
        "lat2d": lat2d,
        "lon2d": lon2d,
        "mod1": mod1.reshape(-1, 3),
        "mod2": mod2.reshape(-1, 3),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "crg_idx": crg_idx,
    }


class SyntheticGeoDataset(Dataset):

    _LEGAL_MODALITIES = {"coords", "mod1", "mod2"}

    def __init__(
        self,
        data: dict[str, torch.Tensor],
        indices: Sequence[int],
        modalities: list[str] = ("coords", "mod1", "mod2"),
    ) -> None:
        """
        PyTorch dataset wrapper for accessing subsets of synthetic geospatial
        modalities in dictionary form.

        Each sample is returned as a dictionary mapping modality names to
        tensors.
        Supported modalities are:
            - "coords" : spatial coordinates (latitude, longitude)
            - "mod1"   : first RGB modality
            - "mod2"   : second RGB modality

        Parameters
        ----------
        data : dict
            Full dataset, where keys are modality names and values are tensors
            of shape `(N, D)` or `(N, ...)` containing all samples.
        indices : list
            Indices specifying which samples from `data` to include in this
            dataset instance (e.g., train, validation, or CRG split).
        modalities : list
            Modalities to include in each returned sample. Must be a subset of
            `{"coords", "mod1", "mod2"}`.

        Attributes
        ----------
        modalities : tuple
            Names of the modalities included in each sample.
        _store : dict
            Internal storage of modality tensors filtered to the given
            `indices`.

        Returns
        -------
        sample : dict[str, torch.Tensor]
            When indexed, returns a dictionary containing only the selected
            modalities for the specified sample index.
        """
        assert set(modalities).issubset(
            self._LEGAL_MODALITIES
        ), f"modalities must be subset of {self._LEGAL_MODALITIES}"
        self.modalities = tuple(modalities)
        self._store = {
            k: v[indices] for k, v in data.items() if k in self.modalities
        }

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return next(iter(self._store.values())).shape[0]

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a single sample from the dataset.
        """
        return {k: v[idx] for k, v in self._store.items()}


class SyntheticGeoDataModule(LightningDataModule):
    """
    Lightning‑friendly wrapper producing:
      • train_dataloader()                   – in‑region train samples
      • val_dataloader() -> [ loader0, loader1 ]
            loader0 = in‑region validation (val_idx)
            loader1 = CRG validation         (crg_idx)
    """

    _SPLIT_KEYS = ("train_idx", "val_idx", "crg_idx")

    def __init__(
        self,
        dataset_name: str = "SyntheticPID",
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        modalities: list[str] = ("coords", "mod1", "mod2"),
        coordinate_predictions: str | None = None,
        morans_I: float = 0.0,
        stats_save_path: str = "svi_data/syntheticPID/",
    ) -> None:
        """
        PyTorch Lightning-compatible data module for synthetic geospatial
        datasets.

        Parameters
        ----------
        dataset_name : str, default="SyntheticPID"
            Name identifier for the dataset instance.
        batch_size : int, default=128
            Batch size for all DataLoaders.
        num_workers : int, default=0
            Number of worker processes for DataLoader.
        pin_memory : bool, default=True
            Whether to pin memory in DataLoaders for faster GPU transfer.
        persistent_workers : bool, default=False
            Whether to keep worker processes alive across DataLoader
            iterations.
        modalities : list of str, default=("coords", "mod1", "mod2")
            Modalities to include in each dataset split. Must be a subset of
            ``{"coords", "mod1", "mod2"}``.
        coordinate_predictions : str or None, default=None
            Optional path or setting for coordinate prediction outputs.
        morans_I : float, default=0.0
            Target Moran's I for modality-specific (blue channel) fields.

        Returns
        -------
        LightningDataModule
            Configured data module with train, validation (in-region and CRG),
            and test DataLoaders for synthetic geospatial data.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        raw_np_data = build_data_dict(N=200, noise_sigma=0.0)
        self.dataset_name = dataset_name
        self.modalities = modalities
        self.coordinate_predictions = coordinate_predictions
        self.raw_np_data = raw_np_data
        self.morans_I = morans_I

        # Convert numpy → torch once
        self._tensor_data = {
            "coords": torch.as_tensor(raw_np_data["coord"]).float(),
            "mod1": torch.as_tensor(raw_np_data["mod1"]).float(),
            "mod2": torch.as_tensor(raw_np_data["mod2"]).float(),
        }

        # Store index arrays for splits
        self._split = {k: raw_np_data[k] for k in self._SPLIT_KEYS}
        self._datasets: dict[str, SyntheticGeoDataset] = {}

        # compute & retain stats (new)
        self._compute_stats()

    def setup(self, stage: str | None = None) -> None:
        """
        Instantiate datasets exactly once per process.
        """
        if self._datasets:  # already built
            return

        self._datasets = {
            "train": SyntheticGeoDataset(
                self._tensor_data,
                self._split["train_idx"],
                self.hparams.modalities,
            ),
            "val_in_region": SyntheticGeoDataset(
                self._tensor_data,
                self._split["val_idx"],
                self.hparams.modalities,
            ),
            "val_crg": SyntheticGeoDataset(
                self._tensor_data,
                self._split["crg_idx"],
                self.hparams.modalities,
            ),
        }

    @staticmethod
    def _norm_factor(dim: int, method: str = "sqrt") -> float:
        """
        Compute normalization factor for a given dimension and method.
        """
        if method == "sqrt":
            return 1.0 / math.sqrt(dim)
        elif method == "linear":
            return 1.0 / float(dim)
        raise ValueError(f"Unknown normalization method: {method}")

    def _compute_stats(self) -> None:
        """
        Compute per-channel mean, standard deviation, and scaling factors
        for the training split and store them in ``self.stats``.

        Needed for SMF training.
        """
        mean_parts, std_parts, scale_parts = [], [], []

        train_idx = self._split["train_idx"]

        for name in self.hparams.modalities:
            t_full = self._tensor_data[name]
            t = t_full[train_idx]

            mean_parts.append(t.mean(dim=0))
            std_parts.append(t.std(dim=0, unbiased=False))

            s_val = self._norm_factor(dim=t.shape[1])
            scale_parts.append(torch.full_like(mean_parts[-1], s_val))

        self.stats = {
            "mean": torch.cat(mean_parts, dim=0),
            "std": torch.cat(std_parts, dim=0),
            "scale": torch.cat(scale_parts, dim=0),
        }
        print(self.stats)
        np.save(
            self.hparams.stats_save_path
            + "mean_vector_syntheticPID.npy",
            self.stats["mean"].numpy(),
        )
        np.save(
            self.hparams.stats_save_path
            + "std_vector_syntheticPID.npy",
            self.stats["std"].numpy(),
        )
        np.save(
            self.hparams.stats_save_path
            + "scaling_vector_syntheticPID.npy",
            self.stats["scale"].numpy(),
        )

    def _make_loader(
        self, key: str, shuffle: bool, *, constant_blue: bool
    ) -> DataLoader:
        """
        Create a DataLoader for a specified dataset split.

        Parameters
        ----------
        key : str
            Split key, e.g., "train", "val_in_region", or "val_crg".
        shuffle : bool
            Whether to shuffle the dataset.
        constant_blue : bool
            If True, apply `_constant_blue_collate` to replace the blue channel
            with a constant value per batch.

        Returns
        -------
        DataLoader
            Configured DataLoader for the specified split.
        """
        collate = _constant_blue_collate if constant_blue else None
        return DataLoader(
            self._datasets[key],
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=collate,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the training split.
        """
        # training: constant blue tint per mini‑batch
        return self._make_loader("train", shuffle=True, constant_blue=True)

    def val_dataloader(self) -> list:
        """
        Create DataLoaders for the validation splits.
        """
        # validation & CRG: KEEP the original noisy/unique blue
        return [
            self._make_loader(
                "val_in_region", shuffle=False, constant_blue=False
            ),
            self._make_loader("val_crg", shuffle=False, constant_blue=False),
        ]

    def test_dataloader(self) -> list:
        """
        Create DataLoaders for the test splits.
        """
        return self.val_dataloader()
