from typing import Generic, Protocol, TypeVar


# Generic type for a container, usually array or table-like. E.g. xarray.Dataset or pandas.DataFrame.
DataContainer = TypeVar("DataContainer")


class PrePostProcessable(Generic[DataContainer], Protocol):
    """
    Object with a preprocessing method and a post-processing method.

    Pre- and post-processing might be used around a model projection or a transformation that reduces
    regularly gridded arrays into regional aggregations.
    """

    def preprocess(self, d: DataContainer) -> DataContainer: ...
    def postprocess(self, d: DataContainer) -> DataContainer: ...


class Projectable(Generic[DataContainer], Protocol):
    """
    Object with a project() method to project effects, impacts, damages, etc.
    """

    def project(self, d: DataContainer) -> DataContainer: ...


class PrePostProjectable(
    Generic[DataContainer],
    PrePostProcessable[DataContainer],
    Projectable[DataContainer],
    Protocol,
):
    """
    Object with methods to preprocess(), project(), and postprocess()
    """

    ...
