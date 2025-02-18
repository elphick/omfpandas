import importlib.metadata
from typing import Union, Optional, Literal

import numpy as np
import pandas as pd

from omfpandas import OMFPandasReader
from omfpandas.blockmodels.factory import blockmodel_to_df_factory, df_to_blockmodel_factory
from omfpandas.blockmodels.geometry import GeometryBase, TensorGeometry, RegularGeometry


class OMFBlockModel:
    def __init__(self, blockmodel: Union['BaseBlockModel', 'RegularBlockModel', 'TensorGridBlockModel']):
        from omfpandas import __omf_version__
        self.omf_version = __omf_version__
        self.blockmodel = blockmodel
        self.bm_type: str = blockmodel.__class__.__name__
        self.geometry: GeometryBase = TensorGeometry.from_element(
            blockmodel) if self.bm_type == 'TensorGridBlockModel' else RegularGeometry.from_element(blockmodel)

    def to_dataframe(self, variables: Optional[list[str]] = None, query: Optional[str] = None,
                     index_filter: Optional[list[int]] = None) -> pd.DataFrame:
        return blockmodel_to_df_factory(is_tensor=self.bm_type == 'TensorGridBlockModel')(blockmodel=self.blockmodel,
                                                                                          variables=variables,
                                                                                          query=query,
                                                                                          index_filter=index_filter)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, blockmodel_name: str,
                       blockmodel_type: Optional[Literal['regular', 'tensor']] = 'regular'):
        if blockmodel_type is None:
            if 'x' not in df.index.names and 'y' not in df.index.names and 'z' not in df.index.names:
                raise ValueError("Dataframe must have centroid coordinates (x, y, z) in the index.")
            elif 'dx' in df.index.names and 'dy' in df.index.names and 'dz' in df.index.names:
                is_tensor: bool = True
            else:
                is_tensor: bool = False
        else:
            is_tensor = True if blockmodel_type == 'tensor' else False
        return cls(blockmodel=df_to_blockmodel_factory(is_tensor=is_tensor)(df=df, blockmodel_name=blockmodel_name))
