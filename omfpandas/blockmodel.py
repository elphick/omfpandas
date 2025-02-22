from typing import Union, Optional

import pandas as pd

from omfpandas.blockmodels.convert_blockmodel import blockmodel_to_df, df_to_blockmodel
from omfpandas.blockmodels.geometry import Geometry, TensorGeometry, RegularGeometry


class OMFBlockModel:
    def __init__(self, blockmodel: Union['BaseBlockModel', 'RegularBlockModel', 'TensorGridBlockModel']):
        from omfpandas import __omf_version__
        self.omf_version = __omf_version__
        self.blockmodel = blockmodel
        self.bm_type: str = blockmodel.__class__.__name__
        self.geometry: Geometry = TensorGeometry.from_element(
            blockmodel) if self.bm_type == 'TensorGridBlockModel' else RegularGeometry.from_element(blockmodel)

    def to_dataframe(self, variables: Optional[list[str]] = None, query: Optional[str] = None,
                     index_filter: Optional[list[int]] = None) -> pd.DataFrame:
        return blockmodel_to_df(blockmodel=self.blockmodel,
                                variables=variables,
                                query=query,
                                index_filter=index_filter)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, blockmodel_name: str):
        return cls(blockmodel=df_to_blockmodel(df=df, blockmodel_name=blockmodel_name))
