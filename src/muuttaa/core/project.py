from typing import Any
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import xarray as xr


def _default_predictors_parameters_merge(x: Iterable[xr.Dataset]) -> xr.Dataset:
    return xr.merge(x)


@dataclass(frozen=True)
class Projector:
    """
    Model to project effects, impacts and/or damages.
    """

    preprocess: Callable[[xr.Dataset], Any]
    project: Callable[[Any], Any]
    postprocess: Callable[[Any], xr.Dataset]


def project(
    predictors: xr.Dataset,
    *,
    model: Projector,
    parameters: xr.Dataset,
    merge_predictors_parameters: Callable[[Iterable[xr.Dataset]], xr.Dataset]
    | None = None,
) -> xr.Dataset:
    """
    Project given predictors, a model, and model parameters.
    """
    if merge_predictors_parameters is None:
        merge_predictors_parameters = _default_predictors_parameters_merge

    # Include model artifacts/params/coefs on input xr.Dataset so broadcasting/index problems are hit early... Also makes testing models easier.
    projection_input = merge_predictors_parameters([predictors, parameters])

    preprocessed = model.preprocess(projection_input)
    projected = model.project(preprocessed)
    postprocessed = model.postprocess(projected)

    return postprocessed


# Think about this vs pickling (or use joblib) a model with parameters into an artifact that gets passed around.

# We can use this same project() + Projector pattern to model impacts and then damages, if needed.
