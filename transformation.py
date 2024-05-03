from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Mapping, Protocol, Any, Optional
from os import PathLike
from io import BufferedIOBase

import xarray as xr
import numpy as np


# TODO: How do we want to apply multiple (but select) strategies to input data?
# TODO: What does this look like if we drop it in hydra (which we could use on HPCs)? Do pangeo images have hydra?
# TODO: Custom exceptions for common things to catch?


class Regionalizer(Protocol):
    """
    Use to regionalize gridded, n-dimentional dataset.

    The key is that it can be loaded with invariants or state information (e.g. weights, spatial geometry) before it is used in the transformation process to regionalize gridded data.
    """

    def regionalize(self, ds: xr.Dataset) -> xr.Dataset:
        ...


# TODO: Can we still read docstr of callables through their attributes after theyve been passed to an instantiated TransformationStrategy?
@dataclass(frozen=True)
class TransformationStrategy:
    """
    Named, tranformation steps applied to input gridded data, pre/post regionalization, to create a derived variable as output.

    These steps should be general. They may contain logic for sanity checks on inputs and outputs, calculating derived variables and climate indices, adding or checking metadata or units. Avoid including logic for cleaning, or harmonizing input data, especially if it is specific to a single project's usecase. Generally avoid using a single strategy to output multiple unrelated variables.
    """

    preprocess: Callable[[xr.Dataset], xr.Dataset]
    postprocess: Callable[[xr.Dataset], xr.Dataset]


# TODO: Add with registration function? More sophistication needed? See pint for e.g.? Do we want people to pass registries around?
# TODO: I fear global variables. We should pass around a strategy registry instead of using globals.
_STRATEGY_REGISTRY: Mapping[str, TransformationStrategy] = {}


def register_strategy(s: TransformationStrategy) -> bool:
    # TODO: Error, warn, or ignore on duplicate key? How does pint do this? Be consistent with something people are familiar with.
    identifier = str(s.identifier)

    # Warn people if they are setting a strategy with a key already in use.
    if identifier in _STRATEGY_REGISTRY:
        import warnings

        warnings.warn(
            f"{identifier=} is already used in the strategy registry but the strategy is being overwritten by a new entry",
            stacklevel=2,  # TODO Might need to change stacklevel. Test this.
        )

    _STRATEGY_REGISTRY[identifier] = s
    return True


# Use class for segment weights because we're making assumptions/enforcements about the weight data's content and interactions...
class SegmentWeights:
    """
    Segment weights to regionalize regularly-gridded data
    """

    def __init__(self, weights: xr.Dataset):
        target_variables = ("lat", "lon", "weight", "region")
        missing_variables = (v for v in target_variables if v not in weights.variables)
        if missing_variables:
            raise ValueError(
                "input weights is missing required {missing_variables} variable(s)"
            )
        self._data = weights

    def regionalize(self, x: xr.Dataset) -> xr.Dataset:
        """
        Regionalize input gridded data
        """
        # TODO: See how this errors in different common scenarios. What happens on the unhappy path?
        region_sel = x.sel(lat=self._data["lat"], lon=self._data["lon"])
        out = (region_sel * self._data["weight"]).groupby(self._data["region"]).sum()
        # TODO: Maybe drop lat/lon and set 'region' as dim/coord? I feel like we can do this because we're asking weights to strictly match input's lat/lon. Maybe make this a req of segment weights we're reading in?
        return out

    def __call__(self, x: xr.Dataset) -> xr.Dataset:
        return self.regionalize(x)


def open_segmentweights(
    weights_url: str | PathLike[Any] | BufferedIOBase,
) -> SegmentWeights:
    """
    Open segment weights from storage or computer magic
    """
    ...


def transform(
    obj: xr.Dataset,
    *,
    strategy: TransformationStrategy,
    regionalize: Callable[[xr.Dataset], xr.Dataset],
) -> xr.Dataset:
    """
    Transform input gridded data to regional data, given a strategy for pre/post processing
    """
    return strategy.postprocess(regionalize(strategy.preprocess(obj)))


def _default_transform_merge(x: Sequence[xr.Dataset]) -> xr.Dataset:
    return xr.merge(x)


# TODO: This regionalizer type is almost repeated enough to create it's own type.
def apply_transformations(
    gridded: xr.Dataset,
    *,
    strategies: Sequence[TransformationStrategy],
    regionalize: Callable[[xr.Dataset], xr.Dataset],
    merge_transformed: Optional[Callable[[Sequence[xr.Dataset]], xr.Dataset]] = None,
) -> xr.Dataset:
    """
    Apply multiple regionalized transformations output to a single Dataset.
    """
    strategies = tuple(strategies)

    if merge_transformed is None:
        merge_transformed = _default_transform_merge

    regionalized_transforms = [
        transform(gridded, regionalize=regionalize, strategy=s) for s in strategies
    ]
    merged = merge_transformed(regionalized_transforms)
    return merged


# ! Make weights easy to swap in and out for the same strategy. Were to pass in weights?

# Use plugin framework so external packages can add own strategies? Do we need that?

# How will people use it? Multiple transforms, multiple weights, creating new transform strategies. Check these usecases.

# Create a registry for TransformationStrategies?

# **How to make strategies useable with BigQuery-generated weights or something with xagg or legacy method?


#############################################################
"""
Testing how we actually regionalize data...
"""

da = xr.DataArray(
    np.arange(25).reshape([5, 5]),
    dims=("x", "y"),
    coords={
        "x": np.arange(5),
        "y": np.arange(5),
    },
)

wgts = xr.Dataset(
    {
        "region": (["idx"], ["a", "a", "a", "b"]),
        "weight": (["idx"], np.array([0.3, 0.3, 0.3, 1.0])),
        "x": (["idx"], [2, 3, 4, 1]),
        "y": (["idx"], [0, 0, 0, 2]),
    },
)

goal = xr.DataArray(
    np.array([13.5, 7.0]),
    dims="region",
    coords={
        "region": ["a", "b"],
    },
)
actual = (
    (da.sel(y=wgts["y"], x=wgts["x"]) * wgts["weight"]).groupby(wgts["region"]).sum()
)

xr.testing.assert_allclose(actual, goal)

# What if we add extra dim (z) to input data. Is this preserved?

da2 = xr.DataArray(
    np.arange(125).reshape([5, 5, 5]),
    dims=("x", "y", "z"),
    coords={
        "x": np.arange(5),
        "y": np.arange(5),
        "z": np.arange(5),
    },
)

actual = (
    (da2.sel(y=wgts["y"], x=wgts["x"]) * wgts["weight"]).groupby(wgts["region"]).sum()
)
# TODO: Define assertion and `goal` to actually test this.
