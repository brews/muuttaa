from sandbox import open_segmentweights, TransformationStrategy, transform
import xarray as xr


SG_URL = "gs://abucket/segweights2024.zarr"
INPUT_DOWNSCALED_URL = "gs://abucket/inputdata.zarr"



def addone(ds):
    ds["tasmax"] += 1
    return ds

def check_ds(ds):
    if (ds < 0).any():
        raise ValueError("We have a value we shouldn't have")
    return ds

our_approach = TransformationStrategy(
    identifier="tasmax_mccusker2024",
    pre_regionalization=addone,
    post_regionalization=check_ds,
)



swgts = open_segmentweights(SG_URL)
ds_in = xr.open_zarr(INPUT_DOWNSCALED_URL)

out = transform(ds_in, strategy=our_approach, region_maker=swgts)