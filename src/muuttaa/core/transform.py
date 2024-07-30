from typing import Protocol
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import xarray as xr


class Regionalizer(Protocol):
    """
    Use to regionalize gridded, n-dimentional dataset.

    The key is that it can be loaded with invariants or state information (e.g. weights, spatial geometry) before it is used in the transformation process to regionalize gridded data.
    """

    def regionalize(self, ds: xr.Dataset) -> xr.Dataset: ...


# TODO: Can we still read docstr of callables through their attributes after theyve been passed to an instantiated TransformationStrategy?
@dataclass(frozen=True)
class TransformationStrategy:
    """
    Named, tranformation steps applied to input gridded data, pre/post regionalization, to create a derived variable as output.

    These steps should be general. They may contain logic for sanity checks on inputs and outputs, calculating derived variables and climate indices, adding or checking metadata or units. Avoid including logic for cleaning, or harmonizing input data, especially if it is specific to a single project's usecase. Generally avoid using a single strategy to output multiple unrelated variables.
    """

    preprocess: Callable[[xr.Dataset], xr.Dataset]
    postprocess: Callable[[xr.Dataset], xr.Dataset]


# Use class for segment weights because we're making assumptions/enforcements about the weight data's content and interactions...
class SegmentWeights:
    """
    Segment weights to regionalize regularly-gridded data
    """

    def __init__(self, weights: xr.Dataset):
        target_variables = ("lat", "lon", "weight", "region")
        missing_variables = [v for v in target_variables if v not in weights.variables]
        if missing_variables:
            raise ValueError(
                f"input weights is missing required {missing_variables} variable(s)"
            )
        self._data = weights

    def regionalize(self, x: xr.Dataset) -> xr.Dataset:
        """
        Regionalize input gridded data
        """
        # TODO: See how this errors in different common scenarios. What happens on the
        #  unhappy path?
        region_sel = x.sel(lat=self._data["lat"], lon=self._data["lon"])
        out = (region_sel * self._data["weight"]).groupby(self._data["region"]).sum()
        # TODO: Maybe drop lat/lon and set 'region' as dim/coord? I feel like we can do
        #  this because we're asking weights to strictly match input's lat/lon. Maybe
        #  make this a req of segment weights we're reading in?
        return out

    def __call__(self, x: xr.Dataset) -> xr.Dataset:
        return self.regionalize(x)


def _default_transform_merge(x: Iterable[xr.Dataset]) -> xr.Dataset:
    return xr.merge(x)


def apply_transformations(
    gridded: xr.Dataset,
    *,
    strategies: Iterable[TransformationStrategy],
    regionalize: Callable[[xr.Dataset], xr.Dataset],
    merge_transformed: Callable[[Iterable[xr.Dataset]], xr.Dataset] | None = None,
) -> xr.Dataset:
    """
    Apply multiple regionalized transformations output to a single Dataset.
    """
    strategies = tuple(strategies)

    if merge_transformed is None:
        merge_transformed = _default_transform_merge

    transformed = []
    for s in strategies:
        preprocessed = s.preprocess(gridded)
        regionalized = regionalize(preprocessed)
        postprocessed = s.postprocess(regionalized)
        transformed.append(postprocessed)

    return merge_transformed(transformed)
