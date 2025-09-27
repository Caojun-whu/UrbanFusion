#!/usr/bin/env python3
"""
Description: Implementation of a PyTorch Dataset and DataModule for the
pretraining encoders for OSM basemap encoding.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from lightning.pytorch import LightningDataModule
from PIL import Image
from torch.utils.data import Dataset


class OSMDataset(Dataset):
    def __init__(self, files: list, transform: callable = None) -> None:
        """
        Custom dataset for loading OSM tile images.

        Parameters
        ----------
        files : list of str
            List of file paths to the images.
        transform : callable, optional
            Transformations to apply to the images.
        """
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            img = Image.open(self.files[idx]).convert("RGB")
        except Exception as e:
            print(f"Error loading image {self.files[idx]}: {e}")
        if self.transform:
            img = self.transform(img)
        return img


class OSMDatasetModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        locations_path: str,
        places_path: str,
        available_images_path: str,
        batch_size: int,
        num_workers: int,
        zoom_levels: list = [15, 16, 17],
        buffer_distances: list = [600, 300, 150],
        out_region_cities: list = ["Milan"],
        in_region_split: float = 0.2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed_split: int = 42,
    ) -> None:
        """
        PyTorch Lightning DataModule for OSM basemap pretraining.

        Parameters
        ----------
        root_dir : str
            Root directory for the dataset.
        locations_path : str
            Path to the locations CSV file.
        places_path : str
            Path to the places CSV file.
        available_images_path : str
            Path to the available images txt file.
        batch_size : int
            Batch size for data loaders.
        num_workers : int
            Number of worker threads for data loading.
        zoom_levels : list of int, optional
            Zoom levels for the images. Default is [15, 16, 17].
        buffer_distances : list of int, optional
            Buffer distances for the images. Default is [600, 300, 150].
        out_region_cities : list of str, optional
            List of cities to be considered out of region. Default is
            ["Milan"].
        in_region_split : float, optional
            Fraction of in-region data to be used for validation. Default is
            0.2.
        pin_memory : bool, optional
            Whether to pin memory for data loaders. Default is True.
        persistent_workers : bool, optional
            Whether to use persistent workers for data loaders. Default is
            True.
        seed_split : int, optional
            Seed for random splitting of in-region data. Default is 42.
        """
        super().__init__()
        self.root_dir = root_dir
        self.locations_path = locations_path
        self.places_path = places_path
        self.available_images_path = available_images_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.zoom_levels = zoom_levels
        self.buffer_distances = buffer_distances
        self.out_region_cities = out_region_cities
        self.in_region_split = in_region_split
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.transform = T.Compose([T.ToTensor()])
        self.seed_split = seed_split

    def setup(self, stage: str = None) -> None:
        """
        Setup the dataset for training or validation.

        Parameters
        ----------
        stage : str, optional
            The stage for which to setup the dataset
            (e.g., "fit", "validate", "test").
        """

        # Load PlacePulse 2.0 data
        df_locations = pd.read_csv(self.locations_path, sep="\t")
        df_locations = df_locations.rename(columns={"_id": "location_id"})
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

        # Assign initial split
        gdf["split"] = gdf["place_name"].apply(
            lambda x: (
                "validation_out_region"
                if x in self.out_region_cities
                else "training"
            )
        )
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
        train_gdf = gdf[gdf["split"] == "training"]
        in_region_gdf = gdf[gdf["split"] == "validation_in_region"]
        out_region_gdf = gdf[gdf["split"] == "validation_out_region"]
        train_city_files = []
        for zoom, buffer in zip(self.zoom_levels, self.buffer_distances):
            for original_name in train_gdf["image_name"]:
                name_base = original_name.replace(".JPG", ".tif")
                new_name = f"{buffer}_{zoom}_{name_base}"
                train_city_files.append(new_name)

        val_out_region_files = []
        for zoom, buffer in zip(self.zoom_levels, self.buffer_distances):
            for original_name in out_region_gdf["image_name"]:
                name_base = original_name.replace(".JPG", ".tif")
                new_name = f"{buffer}_{zoom}_{name_base}"
                val_out_region_files.append(new_name)

        val_in_region_files = []
        for zoom, buffer in zip(self.zoom_levels, self.buffer_distances):
            for original_name in in_region_gdf["image_name"]:
                name_base = original_name.replace(".JPG", ".tif")
                new_name = f"{buffer}_{zoom}_{name_base}"
                val_in_region_files.append(new_name)
        self.train_city_files = train_city_files
        self.val_in_region_files = val_in_region_files
        self.val_out_region_files = val_out_region_files
        print(
            f"Train size: {len(self.train_city_files)}, "
            f"Val size (in region): {len(self.val_in_region_files)}, "
            f"Val size (out region): {len(self.val_out_region_files)}"
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the DataLoader for the training dataset.

        Return
        -------
        torch.utils.data.DataLoader
            DataLoader for the training dataset.
        """
        dataset = OSMDataset(
            files=[f"{self.root_dir}/{f}" for f in self.train_city_files],
            transform=self.transform,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> list:
        """
        Create the DataLoader for the validation dataset.

        Returns
        -------
        list
            List of DataLoaders for the validation dataset.
        """
        in_region_dataset = OSMDataset(
            files=[f"{self.root_dir}/{f}" for f in self.val_in_region_files],
            transform=self.transform,
        )
        out_region_dataset = OSMDataset(
            files=[f"{self.root_dir}/{f}" for f in self.val_out_region_files],
            transform=self.transform,
        )

        loader_random = torch.utils.data.DataLoader(
            in_region_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

        loader_city = torch.utils.data.DataLoader(
            out_region_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

        return [loader_random, loader_city]
