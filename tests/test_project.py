import xarray as xr

from muuttaa import Projector, project


def test_basic_projection():
    """
    Basic test running a Projector with project().
    """
    predictors = xr.Dataset({"foobar": (["idx"], [0, 0, 0])})
    params = xr.Dataset({"ni": (["idx"], [1, 2, 3])})
    expected = xr.Dataset({"impact": (["idx"], [13, 14, 15])})

    def _pre(x):
        out = xr.Dataset()
        out["foobar"] = x["foobar"] + 1
        out["ni"] = x["ni"]
        return out

    def _post(x):
        return x[["impact"]] + 10

    def _model(x):
        return (x["foobar"] * 2 + x["ni"]).to_dataset(name="impact")

    test_impact_model = Projector(
        preprocess=_pre,
        project=_model,
        postprocess=_post,
    )

    actual = project(predictors, model=test_impact_model, parameters=params)
    xr.testing.assert_allclose(actual, expected)
