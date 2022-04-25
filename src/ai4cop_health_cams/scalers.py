from abc import ABC, abstractmethod

import xarray as xr
import numpy as np

from ai4cop_health_cams.logger import get_logger

LOGGER = get_logger(__name__)


class DataTransformer(ABC):
    def __init__(self, name: str, fit: bool = False):
        self.name = name
        self.fit = fit

    def is_fit(self) -> bool:
        """Returns a flag indicating whether the scaler has been fit to data"""
        return self.fit

    @abstractmethod
    def fit_transform(self, data: xr.DataArray) -> xr.DataArray:
        pass

    @abstractmethod
    def transform(self, data: xr.DataArray) -> xr.DataArray:
        pass

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        pass


class SimpleScaler(DataTransformer):
    def __init__(self, name: str, scaling_factor: float = 1.0):
        """Initialize the object.
        Args:
            name: variable name
            scaling_factor: multiplicative scaling factor
        """
        super().__init__(name, fit=True)
        self.scaling_factor = scaling_factor

    def fit_transform(self, data: xr.DataArray) -> xr.DataArray:
        """Returns data * scaling_factor"""
        return data * self.scaling_factor

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        """Same as fit_transform()"""
        return data * self.scaling_factor

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform: 1/scaling_factor * data"""
        return data / self.scaling_factor


class Log1pDataScaler(DataTransformer):
    def __init__(self, name: str, scaling_factor: float = 1.0):
        """Initialize the object.
        Args:
            name: variable name
            scaling_factor: multiplicative scaling factor (applied before the log1p)
        """
        super().__init__(name, fit=True)
        self.scaling_factor = scaling_factor

    def fit_transform(self, data: xr.DataArray) -> xr.DataArray:
        """Returns log1p(data * scaling_factor)"""
        return np.log1p(data * self.scaling_factor)

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        """Same as fit_transform()"""
        return np.log1p(data * self.scaling_factor)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform: exp((scaling_factor ^ -1) * data) - 1"""
        return np.exp(data / self.scaling_factor) - 1.0


class StdScaler(DataTransformer):
    def __init__(self, name: str) -> None:
        super().__init__(name, fit=False)
        self.mean = None
        self.std = None

    def fit_transform(self, data: xr.DataArray) -> xr.DataArray:
        """Normalization along 1st axis"""
        self.mean = data.mean(axis=0).compute()
        self.std = data.std(axis=0).compute()
        self.std = self.std.where(self.std > 1e-5, 1.0)
        self.fit = True
        return (data - self.mean) / self.std

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        assert self.fit, "Fit the scaler first!"
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        assert self.fit, "Fit the scaler first!"
        return data * self.std + self.mean


class MinMaxScaler(DataTransformer):
    def __init__(self, name: str) -> None:
        super().__init__(name, fit=False)
        self.min = None
        self.max = None
        self.eps = 1e-10

    def fit_transform(self, data: xr.DataArray) -> xr.DataArray:
        """Min-max scaling"""
        self.min = data.min(axis=0, skipna=True).compute()
        self.max = data.max(axis=0, skipna=True).compute()
        self.fit = True
        return (data - self.min) / (self.max - self.min + self.eps)

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        assert self.fit, "Fit the scaler first!"
        return (data - self.min) / (self.max - self.min + self.eps)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        assert self.fit, "Fit the scaler first!"
        return data * (self.max - self.min) + self.min


class CubeRootScaler(DataTransformer):
    def __init__(self, name: str, scaling_factor: float = 1.0):
        """Initialize the object.
        Args:
            name: variable name
            scaling_factor: multiplicative scaling factor (applied before the cbrt!)
        """
        super().__init__(name, fit=True)
        self.scaling_factor = scaling_factor

    def fit_transform(self, data: xr.DataArray) -> xr.DataArray:
        """Returns log1p(data * scaling_factor)"""
        return np.cbrt(data * self.scaling_factor)

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        """Same as fit_transform()"""
        return np.cbrt(data * self.scaling_factor)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform: (data/scaling_factor)**3"""
        return np.power(data / self.scaling_factor, 3)


class Log1pStdScaler(DataTransformer):
    def __init__(self, name: str, log1p_mu: np.ndarray, log1p_sd: np.ndarray) -> None:
        """
        Args:
            name: name of transformer object
            log1p_mu, log1p_sd: mean and standard deviation of log1p(global data) (precomputed)
        """
        super().__init__(name, fit=True)
        self.log1p_mu = log1p_mu
        self.log1p_sd = np.where(log1p_sd >= 1e-5, log1p_sd, 1.0)  # guard against division by zero
        assert self.log1p_mu.shape == self.log1p_sd.shape  # sanity check

    def fit_transform(self, data: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError("Call transform() directly!")

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        """Take logs, then standardize."""
        LOGGER.debug("Data shape: %s", data.values.shape)
        LOGGER.debug("Summary stats shapes: %s, %s", self.log1p_mu.shape, self.log1p_sd.shape)
        assert data.shape[1:] == self.log1p_mu.shape[1:]  # sanity check
        # first, take log1p
        data = np.log1p(data)
        # then, standardize
        return (data - self.log1p_mu) / self.log1p_sd

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """The inverse transformation"""
        data = data * self.log1p_sd[None, ...] + self.log1p_mu[None, ...]
        return np.exp(data) - 1.0
