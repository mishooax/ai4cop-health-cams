from typing import List, Tuple, Dict, Optional, Mapping
import xarray as xr
import numpy as np

from torch.utils.data import Dataset
import albumentations as A

from ai4cop_health_cams.scalers import DataTransformer
from ai4cop_health_cams.logger import get_logger

from ai4cop_health_cams.debug import NUMPY_CHECK_FINITE


LOGGER = get_logger(__name__)


class AirQualityDataset(Dataset):
    def __init__(
        self,
        file_path_lr_input: str,
        lr_input_vars: List[str],
        time_idx: Tuple[int, int],
        file_path_hr_output: Optional[str] = None,
        hr_output_vars: Optional[List[str]] = None,
        file_path_hr_input: Optional[str] = None,
        hr_input_vars: Optional[List[str]] = None,
        transformers: Optional[Dict[str, DataTransformer]] = None,
        augmentations: Optional[A.Compose] = None,
        check_finite: bool = False,
        chunk_size_lat_lon: int = 32,
    ) -> None:
        """
        Reads input LR / HR data into memory and scales it as requested.
        Args:
            file_path_lr_input: path to LR input data file
            lr_input_vars: LR input variable names
            time_idx: index time range [start_time, end_time]
            file_path_hr_input: path to HR input data file
            hr_input_vars: HR input variable names ("static" fields: orog, lsm, built_frac, ...)
            file_path_hr_output: path to HR input data file
            hr_output_vars: HR output variable names
            transformers: mapping {"varname" : data transformer object}
            check_finite: check that the batch tensors are free of NaNs and Infs (raises an error if any are detected)
            chunk_size_lat_lon: Dask chunk size for latitude and longitude dimensions
                                (set to -1 if you want a single chunk i.e. no parallelism)
        """
        super().__init__()
        self.check_finite = check_finite

        LOGGER.debug("Reading low-res inputs %s from %s ...", lr_input_vars, file_path_lr_input)
        with xr.open_dataset(file_path_lr_input, chunks={"time": -1}) as ds_lr:
            self.lr_in: xr.Dataset = ds_lr[lr_input_vars].isel(time=slice(*time_idx))
        self.length = len(self.lr_in.time)

        self.hr_out: Optional[xr.Dataset] = None
        if file_path_hr_output is not None and hr_output_vars is not None:
            LOGGER.debug("Reading hi-res outputs %s from %s ...", hr_output_vars, file_path_hr_output)
            with xr.open_dataset(
                file_path_hr_output, chunks={"time": -1, "longitude": chunk_size_lat_lon, "latitude": chunk_size_lat_lon}
            ) as ds_hr:
                self.hr_out = ds_hr[hr_output_vars].isel(time=slice(*time_idx))

        self.hr_in: Optional[xr.Dataset] = None
        if file_path_hr_input is not None and hr_input_vars is not None:
            LOGGER.debug("Reading hi-res constant inputs %s from %s ...", hr_input_vars, file_path_hr_input)
            with xr.open_dataset(file_path_hr_input) as ds_static:
                self.hr_in = ds_static[hr_input_vars]

        self.lr_inputs = lr_input_vars
        self.hr_inputs = hr_input_vars
        self.hr_outputs = hr_output_vars

        # interpolate NaNs (1D linear interpolation)
        self._interpolate_nans()

        # transform (scale) the data
        self.scalers = transformers
        if transformers is not None:
            self._transform()

        self._compute_all()

        self.augmentations = augmentations

    def _transform(self):
        """Transforms the data variables. Uses the self.transformers objects."""
        # LR inputs
        for var in self.lr_inputs:
            if not self.scalers[f"{var}_LR"].is_fit():
                self.lr_in[var] = self.scalers[f"{var}_LR"].fit_transform(self.lr_in[var])
            else:
                self.lr_in[var] = self.scalers[f"{var}_LR"].transform(self.lr_in[var])

        # HR (static) inputs: land-sea mask, orography, building data, ...
        if self.hr_inputs is not None:
            for var in self.hr_inputs:
                if not self.scalers[f"{var}_HR"].is_fit():
                    self.hr_in[var] = self.scalers[f"{var}_HR"].fit_transform(self.hr_in[var])
                else:
                    self.hr_in[var] = self.scalers[f"{var}_HR"].transform(self.hr_in[var])

        # HR outputs (targets)
        if self.hr_outputs is not None:
            for var in self.hr_outputs:
                if not self.scalers[f"{var}_HR"].is_fit():
                    self.hr_out[var] = self.scalers[f"{var}_HR"].fit_transform(self.hr_out[var])
                else:
                    self.hr_out[var] = self.scalers[f"{var}_HR"].transform(self.hr_out[var])

    def _interpolate_nans(self) -> None:
        """Fills NaNs using 1D linear interpolation - over latitude."""
        for var in self.lr_inputs:
            self.lr_in[var] = self.lr_in[var].interpolate_na(dim="time", method="linear", keep_attrs=True)
        if self.hr_out is not None:
            for var in self.hr_outputs:
                self.hr_out[var] = self.hr_out[var].interpolate_na(dim="time", method="linear", keep_attrs=True)
        if self.hr_in is not None:
            for var in self.hr_inputs:
                self.hr_in[var] = self.hr_in[var].interpolate_na(dim="latitude", method="linear", keep_attrs=True)

    def __len__(self):
        """Returns the length of the dataset"""
        return self.length

    def _compute_all(self):
        self.lr_in = self.lr_in.compute()
        if self.hr_out is not None:
            self.hr_out = self.hr_out.compute()
        if self.hr_in is not None:
            self.hr_in = self.hr_in.compute()

    @property
    def num_lr_input_vars(self) -> int:
        """Returns the number of low-res input variables"""
        return len(self.lr_inputs)

    @property
    def num_hr_input_vars(self) -> int:
        """Returns the number of high-res (static) input variables"""
        return len(self.hr_inputs) if self.hr_inputs is not None else 0

    @property
    def num_hr_output_vars(self) -> int:
        """Returns the number of high-res output variables"""
        return len(self.hr_outputs) if self.hr_outputs is not None else 0

    def __getitem__(self, idx) -> Tuple[Mapping[str, np.ndarray], ...]:
        """
        Returns the data sample at index idx.
        Mapping keys:
            "X_lr": low-res inputs
            "X_hr": high-res inputs [optional]
            "y": high-res output [optional]
        """
        X, Y = {}, {}

        X_lr = np.stack([self.lr_in[var].isel(time=idx).values for var in self.lr_inputs], axis=0)
        if self.check_finite:
            NUMPY_CHECK_FINITE(X_lr)
        X["X_lr"] = X_lr.astype(np.float32)

        if self.hr_out is not None:
            y = np.stack([self.hr_out[var].isel(time=idx).values for var in self.hr_outputs], axis=0)
            if self.check_finite:
                NUMPY_CHECK_FINITE(y)
            Y["y"] = y.astype(np.float32)

        if self.hr_in is not None:
            X_hr = np.stack([self.hr_in[var].values for var in self.hr_inputs], axis=0)
            if self.check_finite:
                NUMPY_CHECK_FINITE(X_hr)
            X["X_hr"] = X_hr.astype(np.float32)

        if self.augmentations is not None:
            X, Y = self._data_augmentation(X, Y)

        return X, Y

    def _data_augmentation(self, X, Y) -> Tuple[Mapping[str, np.ndarray], ...]:
        """Augments the data samples (rotations, translations, etc.)"""
        if (self.hr_in is not None) and (self.hr_out is not None):
            augmented_samples = self.augmentations(image=X["X_lr"], X_hr=X["X_hr"], Y=Y["y"])
            X["X_hr"] = augmented_samples["X_hr"]
            Y["y"] = augmented_samples["Y"]
        elif self.hr_in is not None:
            augmented_samples = self.augmentations(image=X["X_lr"], X_hr=X["X_hr"])
            X["X_hr"] = augmented_samples["X_hr"]
        elif self.hr_out is not None:
            augmented_samples = self.augmentations(image=X["X_lr"], Y=Y["y"])
            Y["y"] = augmented_samples["Y"]
        else:
            augmented_samples = self.augmentations(image=X["X_lr"])
        X["X_lr"] = augmented_samples["image"]

        return X, Y
