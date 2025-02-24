import logging
from typing import Union, Optional

import pandas as pd

from omfpandas.blockmodels.convert_blockmodel import blockmodel_to_df, df_to_blockmodel, df_to_pv_structured_grid, \
    df_to_pv_unstructured_grid
from omfpandas.blockmodels.geometry import Geometry, TensorGeometry, RegularGeometry


class OMFBlockModel:
    def __init__(self, blockmodel: Union['BaseBlockModel', 'RegularBlockModel', 'TensorGridBlockModel']):
        self._logger = logging.getLogger(__class__.__name__)
        self.blockmodel = blockmodel
        self.bm_type: str = blockmodel.__class__.__name__
        self.geometry: Geometry = TensorGeometry.from_element(
            blockmodel) if self.bm_type == 'TensorGridBlockModel' else RegularGeometry.from_element(blockmodel)
        self.attributes = {a.name: a.schema.split('.')[-1] for a in self.blockmodel.attributes}

    def to_dataframe(self, variables: Optional[list[str]] = None, query: Optional[str] = None,
                     index_filter: Optional[list[int]] = None) -> pd.DataFrame:
        return blockmodel_to_df(blockmodel=self.blockmodel,
                                variables=variables,
                                query=query,
                                index_filter=index_filter)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, blockmodel_name: str):
        return cls(blockmodel=df_to_blockmodel(df=df, blockmodel_name=blockmodel_name))

    def plot(self, scalar: str, threshold: bool=True, show_edges: bool = True, show_axes: bool=True) -> 'pv.Plotter':
        import pyvista as pv
        if scalar not in self.attributes.keys():
            raise ValueError(f"Column '{scalar}' not found in the OMFBlockModel.")

        # Create a PyVista plotter
        plotter = pv.Plotter()

        mesh = self.get_blocks(attributes=[scalar])

        # Add a thresholded mesh to the plotter
        if threshold:
            plotter.add_mesh_threshold(mesh, scalars=scalar, show_edges=show_edges)
        else:
            plotter.add_mesh(mesh, scalars=scalar, show_edges=show_edges)

        plotter.title = self.blockmodel.name
        if show_axes:
            plotter.show_axes()

        return plotter

    def get_blocks(self, attributes: Optional[list[str]] = None) -> Union['pv.StructuredGrid', 'pv.UnstructuredGrid']:

        if attributes is None:
            attributes = list(self.attributes.keys())
        df = self.to_dataframe(variables=attributes)

        try:
            # Attempt to create a regular grid
            grid = df_to_pv_structured_grid(df)
            self._logger.debug("Created a pv.StructuredGrid.")
        except ValueError:
            # If it fails, create an irregular grid
            grid = df_to_pv_unstructured_grid(df)
            self._logger.debug("Created a pv.UnstructuredGrid.")
        return grid


