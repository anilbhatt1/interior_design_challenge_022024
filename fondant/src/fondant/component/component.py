"""This module defines interfaces which components should implement to be executed by fondant."""
import os
import typing as t
from abc import abstractmethod

import dask
import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, LocalCluster


class BaseComponent:
    """Base interface for each component, specifying only the constructor.

    Args:
        consume: The schema the component should consume
        produces: The schema the component should produce
        **kwargs: The provided user arguments are passed in as keyword arguments
    """

    def __init__(self):
        self.consumes = None
        self.produces = None

    def teardown(self) -> None:
        """Method called after the component has been executed."""


class DaskComponent(BaseComponent):
    """Component built on Dask."""

    def __init__(self, **kwargs):
        super().__init__()

        # don't assume every object is a string
        dask.config.set({"dataframe.convert-string": False})
        # worker.daemon is set to false because creating a worker process in daemon
        # mode is not possible in our docker container setup.
        dask.config.set({"distributed.worker.daemon": False})

        self.dask_client()

    def dask_client(self) -> Client:
        """Initialize the dask client to use for this component."""
        cluster = LocalCluster(
            processes=True,
            n_workers=os.cpu_count(),
            threads_per_worker=1,
        )
        return Client(cluster)


class DaskLoadComponent(DaskComponent):
    """Component that loads data and returns a Dask DataFrame."""

    @abstractmethod
    def load(self) -> dd.DataFrame:
        pass


class DaskTransformComponent(DaskComponent):
    """Component that transforms an incoming Dask DataFrame."""

    @abstractmethod
    def transform(self, dataframe: dd.DataFrame) -> dd.DataFrame:
        """
        Abstract method for applying data transformations to the input dataframe.

        Args:
            dataframe: A Dask dataframe containing the data specified in the `consumes` section
                of the component specification
        """


class DaskWriteComponent(DaskComponent):
    """Component that accepts a Dask DataFrame and writes its contents."""

    @abstractmethod
    def write(self, dataframe: dd.DataFrame) -> None:
        pass


class PandasTransformComponent(DaskComponent):
    """Component that transforms the incoming dataset partition per partition as a pandas
    DataFrame.
    """

    @abstractmethod
    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for applying data transformations to the input dataframe.
        Called once for each partition of the data.

        Args:
            dataframe: A Pandas dataframe containing a partition of the data
        """


Component = t.TypeVar("Component", bound=BaseComponent)
"""Component type which can represents any of the subclasses of BaseComponent"""