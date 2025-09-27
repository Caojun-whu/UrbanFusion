#!/usr/bin/env python3
"""
Description: Implementation of a PyTorch Dataset and DataModule for the
PlacePulse 2 Multimodal (PP2-M) dataset.
"""
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio
import torch
import torchvision.transforms as T
from lightning.pytorch import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from transformers import AutoImageProcessor


class LatLonDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        modalities: list = ["coords"],
    ) -> None:
        """
        Dataset for loading latitude and longitude data from a CSV file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file containing latitude and longitude data.
        lat_col : str, optional
            Name of the column containing latitude values, by default
            'latitude'.
        lon_col : str, optional
            Name of the column containing longitude values, by default
            'longitude'.
        """
        df = pd.read_csv(filepath)
        self.dataset_name = Path(filepath).stem
        self.modalities = modalities

        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(
                f"Expected columns '{lat_col}' and '{lon_col}' in the file."
            )

        self.lat = torch.tensor(df[lat_col].values, dtype=torch.float32)
        self.lon = torch.tensor(df[lon_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.lat)

    def __getitem__(self, idx):
        data = {}
        data["coords"] = torch.tensor(
            [self.lat[idx], self.lon[idx]], dtype=torch.float32
        )
        for modality in self.modalities:
            if modality != "coords":
                data[modality] = torch.zeros(1, dtype=torch.float32)
        return data


class BucketedBatchSampler(Sampler):
    def __init__(
        self,
        mask_list: list,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
    ) -> None:
        """
        Initialize the BucketedBatchSampler. Only samples from the same
        mask are grouped together. Used for training UrbanFusion on
        datasets, where not every location has all modalities available.

        Parameters
        ----------
        mask_list : list
            List of mask-tuples, one per sample.
        batch_size : int
            Size of the batches to create.
        shuffle : bool, optional
            Whether to shuffle within groups and across batches, by default
            True.
        seed : int, optional
            Optional base seed for reproducibility.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.base_seed = seed

        # Group all sample indices by mask
        self.groups = {}
        for idx, m in enumerate(mask_list):
            self.groups.setdefault(tuple(m), []).append(idx)

        # Precompute total num batches
        # (sum over groups of ceil(len(group)/batch_size))
        self.num_batches = sum(
            (len(idxs) + batch_size - 1) // batch_size
            for idxs in self.groups.values()
        )

    def __iter__(self):
        """
        Iterate over the batches of indices.
        """
        # Fresh RNG each epoch
        rng = np.random.default_rng(self.base_seed)

        all_batches = []
        # For each group: shuffle & chunk
        for idxs in self.groups.values():
            idxs = idxs.copy()
            if self.shuffle:
                idxs = rng.permutation(idxs).tolist()
            for i in range(0, len(idxs), self.batch_size):
                chunk = idxs[i : i + self.batch_size]
                all_batches.append(chunk)

        # Shuffle the order of batches
        if self.shuffle:
            rng.shuffle(all_batches)

        yield from all_batches

    def __len__(self):
        """
        Returns the number of batches.
        """
        # Constant every epoch
        return self.num_batches


class PlacePulse2(Dataset):
    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        modalities: list,
        transform_SVI: callable,
        transform_sentinel2: callable,
        transform_OSM: callable,
        return_filename: bool = False,
        return_coords: bool = False,
        precomputed_h5_path: str = None,
        ablation_masking: str = None,
    ) -> None:
        """
        Custom dataset for PlacePulse 2 Multimodal (PP2-M) dataset.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing the sample data from PlacePulse 2 dataset.
        modalities : list
            List of modalities to include in the dataset.
        transform_SVI : callable
            Transformations to apply to the SVI images.
        transform_sentinel2 : callable
            Transformations to apply to the Sentinel-2 images.
        transform_OSM : callable
            Transformations to apply to the OSM images.
        return_filename : bool, optional
            Whether to return the filename of the SVI image.
        return_coords : bool, optional
            Whether to return the original coordinates of the sample.
        precomputed_h5_path : str, optional
            Path to the precomputed HDF5 file containing the data.
        ablation_masking : str, optional
            Type of ablation masking to apply, only for ablation study,
            defaults to None.
        """
        super().__init__()
        self.gdf = gdf
        self.modalities = modalities
        self.transform_SVI = transform_SVI
        self.transform_sentinel2 = transform_sentinel2
        self.transform_OSM = transform_OSM
        self.return_filename = return_filename
        self.return_coords = return_coords
        self.precomputed_h5_path = precomputed_h5_path
        self.h5 = None
        self.ablation_masking = ablation_masking

    def __getstate__(self) -> dict:
        """
        Custom __getstate__ method to handle the serialization of the dataset.
        This is necessary to avoid issues with the h5py file handle when
        pickling the dataset.

        Returns
        -------
        dict
            The state of the dataset object, excluding the h5py file handle.
        """
        state = self.__dict__.copy()
        state["h5"] = None
        return state

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.gdf)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns a sample from the dataset.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        dict
            A dictionary containing the sample data.
        """
        row = self.gdf.iloc[idx]
        data = {}

        # Use precomputed features if available
        if self.precomputed_h5_path is not None:
            h5_index = row.get("h5_index")
            if self.h5 is None:
                self.h5 = h5py.File(self.precomputed_h5_path, "r")
                if "coords" in self.modalities:
                    self.h5_coords = self.h5["coords"]
                if "SVI" in self.modalities:
                    self.h5_svi = self.h5["SVI"]
                if "sentinel2" in self.modalities:
                    self.h5_sentinel2 = self.h5["sentinel2"]
                if "OSM" in self.modalities:
                    self.h5_osm = self.h5["OSM"]
                if "POI" in self.modalities:
                    self.h5_poi = self.h5["POI"]
            if "coords" in self.modalities:
                pretrained_coords = torch.tensor(
                    self.h5_coords[h5_index], dtype=torch.float32
                )
                original_coords = torch.tensor(
                    [row["loc.0"], row["loc.1"]], dtype=torch.float32
                )
                data["coords"] = torch.cat(
                    (pretrained_coords, original_coords), dim=-1
                )
            if "SVI" in self.modalities:
                data["SVI"] = torch.tensor(
                    self.h5_svi[h5_index], dtype=torch.float32
                )
            if "sentinel2" in self.modalities:
                data["sentinel2"] = torch.tensor(
                    self.h5_sentinel2[h5_index], dtype=torch.float32
                )
            if "OSM" in self.modalities:
                data["OSM"] = torch.tensor(
                    self.h5_osm[h5_index], dtype=torch.float32
                )
            if "POI" in self.modalities:
                data["POI"] = torch.tensor(
                    self.h5_poi[h5_index], dtype=torch.float32
                )
            if self.ablation_masking is not None:
                mask = set(row["modality_mask"])
                data = {
                    mod: (
                        data[mod]
                        if mod in mask
                        else torch.full_like(data[mod], float("nan"))
                    )
                    for mod in self.modalities
                }

        # Use raw data if precomputed features are not available
        else:
            if "coords" in self.modalities:
                coords = torch.tensor(
                    [row["loc.0"], row["loc.1"]], dtype=torch.float32
                )
                data["coords"] = coords
            if "SVI" in self.modalities:
                svi = Image.open(row["image_name"]).convert("RGB")
                svi = self.transform_SVI(svi)
                data["SVI"] = svi
            if "sentinel2" in self.modalities:
                with rasterio.open(row["sentinel2_name"]) as src:
                    sentinel2 = src.read()  # shape: (bands, H, W)
                    sentinel2 = sentinel2 / 10000.0
                    B10 = np.zeros(
                        (1, *sentinel2.shape[1:]), dtype=sentinel2.dtype
                    )
                    sentinel2 = np.concatenate(
                        [sentinel2[:10], B10, sentinel2[10:]], axis=0
                    )
                    sentinel2 = torch.from_numpy(sentinel2).float()
                    sentinel2 = self.transform_sentinel2(sentinel2)
                data["sentinel2"] = sentinel2
            if "OSM" in self.modalities:
                osm_150_17 = Image.open(row["OSM_150_17_name"]).convert("RGB")
                osm_150_17 = self.transform_OSM(
                    images=osm_150_17, return_tensors="pt"
                ).pixel_values.squeeze(0)
                osm_300_16 = Image.open(row["OSM_300_16_name"]).convert("RGB")
                osm_300_16 = self.transform_OSM(
                    images=osm_300_16, return_tensors="pt"
                ).pixel_values.squeeze(0)
                osm_600_15 = Image.open(row["OSM_600_15_name"]).convert("RGB")
                osm_600_15 = self.transform_OSM(
                    images=osm_600_15, return_tensors="pt"
                ).pixel_values.squeeze(0)
                osm = torch.cat((osm_150_17, osm_300_16, osm_600_15), dim=0)
                data["OSM"] = osm
            if "POI" in self.modalities:
                poi = row["text"]
                data["text"] = poi
        if self.return_filename:
            data["gsv_img"] = row["gsv_img"]
        if self.return_coords:
            data["coords_original"] = torch.tensor(
                [row["loc.0"], row["loc.1"]], dtype=torch.float32
            )

        return data


class PlacePulse2DatasetModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        locations_path: str,
        places_path: str,
        available_images_path: str,
        SVI_path: str,
        sentinel2_path: str,
        POI_path: str,
        OSM_basemap_path: str,
        batch_size: int,
        num_workers: int,
        zoom_levels: list = [15, 16, 17],
        buffer_distances: list = [600, 300, 150],
        out_region_cities: list = ["Milan"],
        in_region_split: float = 0.2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed_split: int = 42,
        transform_SVI: callable = None,
        transform_sentinel2: callable = None,
        transform_OSM: callable = None,
        return_filename: bool = False,
        return_coords: bool = False,
        precomputed_h5_path: str = None,
        filter_available_images: bool = False,
        h5_indices_path: str = None,
        save_gdf_path: str = None,
        coordinate_predictions: list = None,
        ablation_masking: str = None,
        modalities: list = ["coords", "SVI", "sentinel2", "OSM", "POI"],
    ) -> None:
        """
        Initialize the PlacePulseDatasetModule for PlacePulse 2 Multimodal
        (PP2-M) dataset.

        Parameters
        ----------
        locations_path : str
            Path to the locations CSV file.
        places_path : str
            Path to the places CSV file (from original Place Pulse 2.0
            Dataset).
        available_images_path : str
            Path to the available images CSV file (from original Place Pulse
            2.0 Dataset).
        SVI_path : str
            Path to the SVI images directory.
        sentinel2_path : str
            Path to the Sentinel-2 images directory.
        POI_path : str
            Path to the POI CSV file.
        OSM_basemap_path : str
            Path to the OSM basemap images directory.
        batch_size : int
            Batch size for the DataLoader.
        num_workers : int
            Number of worker threads for the DataLoader.
        zoom_levels : list, optional
            List of zoom levels for the OSM images, by default [15, 16, 17].
        buffer_distances : list, optional
            List of buffer distances for the OSM images, by default [600, 300,
            150].
        out_region_cities : list, optional
            List of cities for out-of-region validation, by default ["Milan"].
        in_region_split : float, optional
            Fraction of in-region data to use for validation, by default 0.2.
        pin_memory : bool, optional
            Whether to pin memory in the DataLoader, by default True.
        persistent_workers : bool, optional
            Whether to use persistent workers in the DataLoader, by default
            True.
        seed_split : int, optional
            Seed for random splitting of in-region data, by default 42.
        transform_SVI : callable, optional
            Transformations to apply to the SVI images, by default None.
        transform_sentinel2 : callable, optional
            Transformations to apply to the Sentinel-2 images, by default None.
        transform_OSM : callable, optional
            Transformations to apply to the OSM images, by default None.
        return_filename : bool, optional
            Whether to return the filename of the SVI image, by default False.
        return_coords : bool, optional
            Whether to return the original coordinates of the sample, by
            default False.
        precomputed_h5_path : str, optional
            Path to the precomputed HDF5 file containing the data, by default
            None.
        filter_available_images : bool, optional
            Whether to filter the dataset to only include available images, by
            default False.
        h5_indices_path : str, optional
            Path to the HDF5 file containing indices for precomputed data,
            by default None.
        save_gdf_path : str, optional
            Path to save the GeoDataFrame after processing, by default None.
        coordinate_predictions : list, optional
            List of coordinates for generating and saving predictions, by
            default None.
        ablation_masking : str, optional
            Type of ablation masking to apply, only for ablation study,
            defaults to None.
        modalities : list, optional
            List of modalities to include in the dataset, by default
            ["coords", "SVI", "sentinel2", "OSM", "POI"].
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.locations_path = locations_path
        self.places_path = places_path
        self.available_images_path = available_images_path
        self.SVI_path = SVI_path
        self.sentinel2_path = sentinel2_path
        self.POI_path = POI_path
        self.OSM_basemap_path = OSM_basemap_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.zoom_levels = zoom_levels
        self.buffer_distances = buffer_distances
        self.out_region_cities = out_region_cities
        self.in_region_split = in_region_split
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed_split = seed_split
        self.precomputed_h5_path = precomputed_h5_path
        self.filter_available_images = filter_available_images
        self.h5_indices_path = h5_indices_path
        self.save_gdf_path = save_gdf_path
        self.ablation_masking = ablation_masking
        if transform_SVI is None:
            self.transform_SVI = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                    ),
                ]
            )
        else:
            self.transform_SVI = transform_SVI
        if transform_sentinel2 is None:
            # Define the augmentation pipeline
            self.transform_sentinel2 = T.Compose(
                [
                    T.CenterCrop(224),
                ]
            )
        else:
            self.transform_sentinel2 = transform_sentinel2
        if transform_OSM is None:
            self.transform_OSM = AutoImageProcessor.from_pretrained(
                "facebook/vit-mae-base", use_fast=True
            )
        else:
            self.transform_OSM = transform_OSM
        self.return_filename = return_filename
        self.return_coords = return_coords
        self.modalities = modalities
        self.coordinate_predictions = coordinate_predictions

    def setup(self, stage: str = None) -> None:
        """
        Setup the dataset for training or validation.

        Parameters
        ----------
        stage : str, optional
            The stage for which to setup the dataset (train/val/test).
        """
        # Load Place Pulse 2.0 Dataset, merge tables, and preprocess
        df_locations = pd.read_csv(self.locations_path, sep="\t")
        df_locations = df_locations.rename(columns={"_id": "location_id"})
        df_poi = pd.read_csv(self.POI_path)
        df_places = pd.read_csv(self.places_path, sep="\t")
        df_places = df_places.rename(columns={"_id": "place_id"})
        df_all = pd.merge(df_locations, df_places, on="place_id", how="left")
        gdf = gpd.GeoDataFrame(
            df_all,
            geometry=gpd.points_from_xy(df_all["loc.1"], df_all["loc.0"]),
            crs="EPSG:4326",
        )
        gdf["image_name"] = (
            gdf["loc.0"].map(lambda x: f"{x:.6f}")
            + "_"
            + gdf["loc.1"].map(lambda x: f"{x:.6f}")
            + "_"
            + gdf["location_id"].astype(str)
            + "_"
            + gdf["place_name"].str.replace(" ", "", regex=False)
            + ".JPG"
        )
        gdf["gsv_img"] = gdf["image_name"].apply(lambda x: x.split(".JPG")[0])
        gdf["sentinel2_name"] = (
            gdf["loc.0"].map(lambda x: f"{x:.6f}")
            + "_"
            + gdf["loc.1"].map(lambda x: f"{x:.6f}")
            + "_"
            + gdf["location_id"].astype(str)
            + "_"
            + gdf["place_name"].str.replace(" ", "", regex=False)
            + ".tif"
        )
        gdf["OSM_150_17_name"] = (
            "150_17_"
            + gdf["loc.0"].map(lambda x: f"{x:.6f}")
            + "_"
            + gdf["loc.1"].map(lambda x: f"{x:.6f}")
            + "_"
            + gdf["location_id"].astype(str)
            + "_"
            + gdf["place_name"].str.replace(" ", "", regex=False)
            + ".tif"
        )
        gdf["OSM_300_16_name"] = (
            "300_16_"
            + gdf["loc.0"].map(lambda x: f"{x:.6f}")
            + "_"
            + gdf["loc.1"].map(lambda x: f"{x:.6f}")
            + "_"
            + gdf["location_id"].astype(str)
            + "_"
            + gdf["place_name"].str.replace(" ", "", regex=False)
            + ".tif"
        )
        gdf["OSM_600_15_name"] = (
            "600_15_"
            + gdf["loc.0"].map(lambda x: f"{x:.6f}")
            + "_"
            + gdf["loc.1"].map(lambda x: f"{x:.6f}")
            + "_"
            + gdf["location_id"].astype(str)
            + "_"
            + gdf["place_name"].str.replace(" ", "", regex=False)
            + ".tif"
        )
        gdf = pd.merge(gdf, df_poi, on="image_name", how="left")

        available_images_df = pd.read_csv(
            self.available_images_path, sep="\t", header=None
        )
        available_images_df["image_name"] = available_images_df[0].apply(
            lambda x: x.split("/")[-1]
        )
        gdf = gdf.merge(
            available_images_df, on="image_name", how="left", indicator=True
        )
        gdf["image_available"] = gdf["_merge"].apply(
            lambda x: 1 if x == "both" else 0
        )
        gdf = gdf.drop(columns=["_merge"])
        gdf = gdf.drop(columns=[0])
        gdf = gdf[gdf["image_available"] == 1]
        gdf = gdf.drop(columns=["image_available"])

        # Get cities for Cross-Regional Generalization
        gdf["split"] = gdf["place_name"].apply(
            lambda x: (
                "validation_out_region"
                if x in self.out_region_cities
                else "training"
            )
        )
        column_to_basepath = {
            "image_name": self.SVI_path,
            "sentinel2_name": self.sentinel2_path,
            "OSM_150_17_name": self.OSM_basemap_path,
            "OSM_300_16_name": self.OSM_basemap_path,
            "OSM_600_15_name": self.OSM_basemap_path,
        }
        for col, base in column_to_basepath.items():
            base_path = Path(base)
            # apply Path / filename for each, then cast back to str
            gdf[col] = (
                gdf[col]
                .astype(str)  # make sure it’s a string
                .apply(lambda fn: str(base_path / fn))
            )
        # Filter rows where any required file is missing
        required_files = [
            "image_name",
            "sentinel2_name",
            "OSM_150_17_name",
            "OSM_300_16_name",
            "OSM_600_15_name",
        ]

        # If not all samples are available
        if self.filter_available_images:
            for file_col in required_files:
                gdf = gdf[gdf[file_col].apply(lambda x: Path(x).is_file())]

        # Split dataset for training and validation
        training_cities = gdf[gdf["split"] == "training"][
            "place_name"
        ].unique()
        for city in training_cities:
            city_mask = (gdf["place_name"] == city) & (
                gdf["split"] == "training"
            )
            city_indices = gdf[city_mask].index
            n_validation = max(
                1, int(self.in_region_split * len(city_indices))
            )  # At least 1 sample
            rng = np.random.default_rng(seed=self.seed_split)
            validation_indices = rng.choice(
                city_indices, size=n_validation, replace=False
            )
            gdf.loc[validation_indices, "split"] = "validation_in_region"
        rng = np.random.default_rng(seed=self.seed_split)
        gdf = gdf.sample(frac=1, random_state=rng).reset_index(drop=True)
        if self.h5_indices_path is not None:
            h5_indices = pd.read_csv(self.h5_indices_path)
            gdf = gdf.merge(h5_indices, on="gsv_img", how="left")

        self.train_gdf = gdf[gdf["split"] == "training"]
        self.in_region_gdf = gdf[gdf["split"] == "validation_in_region"]
        self.out_region_gdf = gdf[gdf["split"] == "validation_out_region"]

        # Ablation study for training on incomplete modalities
        if self.ablation_masking == "uniform":
            rng_mask = np.random.default_rng(self.seed_split)
            # always include coords, plus choose 1–4 of the remaining
            # modalities
            other_mods = [m for m in self.modalities if m != "coords"]
            masks = []
            for _ in range(len(self.train_gdf)):
                k = int(rng_mask.integers(1, len(other_mods) + 1))
                chosen = list(
                    rng_mask.choice(other_mods, size=k, replace=False)
                )
                masks.append(["coords"] + chosen)
            self.train_gdf["modality_mask"] = masks
            # print counts of how many locations have 2,3,4,5 modalities
            counts = (
                pd.Series([len(m) for m in masks]).value_counts().sort_index()
            )
            print("Modalities‑per‑location after masking:")
            for num_mods, cnt in counts.items():
                print(f"  {num_mods:>1d} modalities: {cnt}")
        elif self.ablation_masking == "distinct":
            # One‑modality ablation: coords + exactly one other modality
            rng_mask = np.random.default_rng(self.seed_split)
            other_mods = [m for m in self.modalities if m != "coords"]
            masks = []
            for _ in range(len(self.train_gdf)):
                # pick exactly one other modality
                chosen = rng_mask.choice(
                    other_mods, size=1, replace=False
                ).tolist()
                masks.append(["coords"] + chosen)
            self.train_gdf["modality_mask"] = masks
            # print counts of how many times each solo modality appears
            flat = [m[1] for m in masks]  # the picked modality per row
            counts = pd.Series(flat).value_counts()
            print("Single‑modality ablation counts (coords + one):")
            for mod, cnt in counts.items():
                print(f"  {mod:>10s}: {cnt}")

        self.gdf = gdf
        print(
            f"Train size: {len(self.train_gdf)}, "
            f"Val size (in region): {len(self.in_region_gdf)}, "
            f"Val size (out region): {len(self.out_region_gdf)}, "
            f"Total size: {len(self.gdf)}"
        )
        if self.save_gdf_path is not None:
            self.gdf.to_csv(self.save_gdf_path, index=False)
            print(f"Saved GeoDataFrame to {self.save_gdf_path}")

        # Setup coordinate dataloaders, save representations for list of
        # locations, used for housing price and energy consumption dataset.
        if self.coordinate_predictions is not None:
            self.coordinate_dataloaders = []
            for coords in self.coordinate_predictions:
                coords_dataset = LatLonDataset(
                    filepath=coords,
                    lat_col="latitude",
                    lon_col="longitude",
                    modalities=self.modalities,
                )
                coords_loader = DataLoader(
                    coords_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    persistent_workers=self.persistent_workers,
                )
                self.coordinate_dataloaders.append(coords_loader)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the training dataloader.
        """
        dataset = PlacePulse2(
            gdf=self.train_gdf,
            modalities=self.modalities,
            transform_SVI=self.transform_SVI,
            transform_sentinel2=self.transform_sentinel2,
            transform_OSM=self.transform_OSM,
            return_filename=self.return_filename,
            return_coords=self.return_coords,
            precomputed_h5_path=self.precomputed_h5_path,
            ablation_masking=self.ablation_masking,
        )

        # Ablation study for training on incomplete modalities
        if self.ablation_masking is None:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )
        mask_list = [
            tuple(self.train_gdf.iloc[i]["modality_mask"])
            for i in range(len(self.train_gdf))
        ]
        sampler = BucketedBatchSampler(
            mask_list=mask_list,
            batch_size=self.batch_size,
            shuffle=True,
            seed=self.seed_split,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> list:
        """
        Create the validation dataloaders.
        """
        in_region_dataset = PlacePulse2(
            gdf=self.in_region_gdf,
            modalities=self.modalities,
            transform_SVI=self.transform_SVI,
            transform_sentinel2=self.transform_sentinel2,
            transform_OSM=self.transform_OSM,
            return_filename=self.return_filename,
            return_coords=self.return_coords,
            precomputed_h5_path=self.precomputed_h5_path,
        )
        out_region_dataset = PlacePulse2(
            gdf=self.out_region_gdf,
            modalities=self.modalities,
            transform_SVI=self.transform_SVI,
            transform_sentinel2=self.transform_sentinel2,
            transform_OSM=self.transform_OSM,
            return_filename=self.return_filename,
            return_coords=self.return_coords,
            precomputed_h5_path=self.precomputed_h5_path,
        )

        in_region_loader = torch.utils.data.DataLoader(
            in_region_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

        out_region_loader = torch.utils.data.DataLoader(
            out_region_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

        return [in_region_loader, out_region_loader]

    def test_dataloader(self) -> list:
        """
        Create the test dataloaders.
        Same as validation dataloaders.
        """
        in_region_dataset = PlacePulse2(
            gdf=self.in_region_gdf,
            modalities=self.modalities,
            transform_SVI=self.transform_SVI,
            transform_sentinel2=self.transform_sentinel2,
            transform_OSM=self.transform_OSM,
            return_filename=self.return_filename,
            return_coords=self.return_coords,
            precomputed_h5_path=self.precomputed_h5_path,
        )
        out_region_dataset = PlacePulse2(
            gdf=self.out_region_gdf,
            modalities=self.modalities,
            transform_SVI=self.transform_SVI,
            transform_sentinel2=self.transform_sentinel2,
            transform_OSM=self.transform_OSM,
            return_filename=self.return_filename,
            return_coords=self.return_coords,
            precomputed_h5_path=self.precomputed_h5_path,
        )

        in_region_loader = torch.utils.data.DataLoader(
            in_region_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

        out_region_loader = torch.utils.data.DataLoader(
            out_region_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

        return [in_region_loader, out_region_loader]

    def predict_dataloader(self) -> DataLoader:
        """
        Return the predict dataloader.

        Returns:
        --------
        DataLoader
            Predict dataloader.
        """
        in_region_dataset = PlacePulse2(
            gdf=self.in_region_gdf,
            modalities=self.modalities,
            transform_SVI=self.transform_SVI,
            transform_sentinel2=self.transform_sentinel2,
            transform_OSM=self.transform_OSM,
            return_filename=self.return_filename,
            return_coords=self.return_coords,
            precomputed_h5_path=self.precomputed_h5_path,
        )
        in_region_loader = torch.utils.data.DataLoader(
            in_region_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return in_region_loader

    def all_locations_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the dataloader for all locations.
        """
        dataset = PlacePulse2(
            gdf=self.gdf,
            modalities=self.modalities,
            transform_SVI=self.transform_SVI,
            transform_sentinel2=self.transform_sentinel2,
            transform_OSM=self.transform_OSM,
            return_filename=self.return_filename,
            return_coords=self.return_coords,
            precomputed_h5_path=self.precomputed_h5_path,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
