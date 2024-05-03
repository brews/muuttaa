import xarray as xr

from projection import ProjectionSpec, Projector, project
from transformation import TransformationStrategy, apply_transformations, open_segmentweights

# Going for explicit. Not magic. Extendable and composible but not too much abstraction.

# Input URLS

downscaled_cmip6_url = "downscaled_cmip6.zarr"
segment_weights_url = "segment_weights.zarr"
impacts_params_url = "impact_model_params.zarr"
valuation_params_url = "valuation_model_params.zarr"

# Define stuff for transforming and regionalizing downscaled_cmip6 input.

downscaled_cmip6 = xr.open_dataset(downscaled_cmip6_url, engine="zarr", chunks={})
segment_weights = open_segmentweights(segment_weights_url)


def do_nothing(ds: xr.Dataset) -> xr.Dataset:
    return ds


def add_one(ds: xr.Dataset) -> xr.Dataset:
    """Revolutionary!"""
    return ds + 1


example_transformation = TransformationStrategy(
    preprocess=do_nothing,
    postprocess=do_nothing
)
another_example_transformation = TransformationStrategy(
    preprocess=add_one,
    postprocess=do_nothing
)

# Define stuff for projecting impact.

impact_model_params = xr.open_dataset(
    impacts_params_url, engine="zarr", chunks={}
)

# Note parameters and predictors are pooled together in the same dataset.
# This is to hit coord and broadcast problems as early and quickly as we can.
def calc_impact(ds: xr.Dataset) -> xr.Dataset:
    out = ds["tasmax"] * ds["beta1"]
    return out.to_dataset(name="impact")

impact_model = Projector(
    preprocess=do_nothing, 
    project=calc_impact,
    postprocess=lambda ds: ds # Another way of saying "do nothing".
)

# Define stuff for valuation model.


def calc_damages(ds: xr.Dataset) -> xr.Dataset:
    out = ds["impact"] * ds["money_metric"]
    return out.to_dataset(name="damages")

# Not sure I like ProjectionSpec. We don't require it here. Just trying to feel it out.
valuation_spec = ProjectionSpec(
    model=Projector(
        preprocess=do_nothing, 
        project=calc_damages, 
        postprocess=do_nothing
    ),
    parameters=xr.open_dataset(valuation_params_url, engine="zarr", chunks={}),
)

##################################################################################
# Now put it all together.

# In input data was Datatree (of downscaled ensemble) could run all this below on all ensemble members with `map_over_subtree()`.

# This transform step likely will have lots of errors and rechunking, plus implementation heavy details about
# how all the transformas are merged together.
transformed = apply_transformations(
    downscaled_cmip6,
    regionalize=segment_weights,
    strategies=[
        example_transformation, 
        another_example_transformation
    ],
    merge_transformed=lambda x: xr.merge(x),  # <- Optional, where we'd define chunks and merge for all the transformed stuff.
)

impacts = project(
    transformed,
    model=impact_model, 
    parameters=impact_model_params
)

# TODO: What about "rebasing"?

damages = project(
    impacts, 
    model=valuation_spec.model, 
    parameters=valuation_spec.parameters
)

# TODO: GCM ensemble weighting?
