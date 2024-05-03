from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Any

# from typing import Protocol

import xarray as xr


# TODO: Protocol or dataclass? I feel like the model artifact makes it stateful with invariants...?
# class Projector(Protocol):
#     """
#     Model to project impacts and/or damages
#     """
#     def load(self, artifacts_uri:  str | PathLike[Any] | BufferedIOBase) -> None:  # TODO: Separate function or not? What's best for testing? Ease of use?
#         """
#         Load artifacts for the model doing the projection.
#         """
#         ...

#     def preprocess(self, projection_input: xr.Dataset) -> Any:
#         ...

#     def project(self, instances: Any) -> Any:
#         """
#         Use model to project instances.
#         """
#         ...

#     def postprocess(self, projection_results: Any) -> xr.Dataset:
#         ...


@dataclass(frozen=True)
class Projector:
    """
    Model to project impacts and/or damages
    """

    preprocess: Callable[[xr.Dataset], Any]
    project: Callable[[Any], Any]
    postprocess: Callable[[Any], xr.Dataset]


# TODO: Not sure we need this.
@dataclass
class ProjectionSpec:
    model: Projector
    parameters: xr.Dataset

# def project_models(projection_input: xr.Dataset, model_artifacts: xr.Dataset, models: Sequence[Projector]):
#     # Not sure this works because we'd have to merge all projection output into a single xr.Dataset.
#     ...


def project(
    predictors: xr.Dataset,
    *,
    model: Projector,
    parameters: xr.Dataset,
) -> xr.Dataset:
    """
    Project given predictors, a model, and model parameters.
    """
    # Include model artifacts/params/coefs on input xr.Dataset so broadcasting/index problems are hit early... Also makes testing models easier.
    projection_input = xr.merge([predictors, parameters])
    out = model.postprocess(model.project(model.preprocess(projection_input)))
    return out

# Think about this vs pickling (or use joblib) a model with parameters into an artifact that gets passed around.

# We can use this same project() + Projector pattern to model impacts and then damages, if needed.
