from typing import Optional, TypeVar, Union

import numpy as np
import pandas as pd
from omf import NumericAttribute, CategoryAttribute
from omf.blockmodel import BaseBlockModel, TensorGridBlockModel

from omfpandas.blockmodels.geometry import TensorGeometry
from omfpandas.blockmodels.v2.attributes import series_to_attribute, read_blockmodel_attributes

# generic type variable, used for type hinting, to indicate that the type is a subclass of BaseBlockModel
BM = TypeVar('BM', bound=BaseBlockModel)


def tensor_bm_to_df(blockmodel: BM,
                    variables: Optional[list[str]] = None,
                    query: Optional[str] = None,
                    index_filter: Optional[list[int]] = None) -> pd.DataFrame:
    """Convert tensor block model to a DataFrame.

    Args:
        blockmodel (BlockModel): The BlockModel to convert.
        variables (Optional[list[str]]): The variables to include in the DataFrame. If None, all variables are included.
        query (Optional[str]): The query to filter the DataFrame.
        index_filter (Optional[list[int]]): List of integer indices to filter the DataFrame.

    Returns:
        pd.DataFrame: The DataFrame representing the BlockModel.
    """
    # read the data
    df: pd.DataFrame = read_blockmodel_attributes(blockmodel, attributes=variables, query=query,
                                                  index_filter=index_filter)
    return df


def df_to_tensor_bm(df: pd.DataFrame, blockmodel_name: str) -> BM:
    """Write a DataFrame to a BlockModel.

    Args:
        df (pd.DataFrame): The DataFrame to convert to a BlockModel.
        blockmodel_name (str): The name of the BlockModel.
        created.

    Returns:
        BlockModel: The BlockModel representing the DataFrame.
    """

    # Sort the dataframe to align with the omf spec
    df.sort_index(level=['z', 'y', 'x'], inplace=True)

    # Create the blockmodel and geometry

    # if is_tensor:
    geometry: TensorGeometry = TensorGeometry.from_multi_index(df.index)

    blockmodel: BM = TensorGridBlockModel(name=blockmodel_name)
    # assign the geometry properties
    blockmodel.corner = geometry.corner
    blockmodel.axis_u = geometry.axis_u
    blockmodel.axis_v = geometry.axis_v
    blockmodel.axis_w = geometry.axis_w
    blockmodel.tensor_u = geometry.tensor_u
    blockmodel.tensor_v = geometry.tensor_v
    blockmodel.tensor_w = geometry.tensor_w

    # add the data
    attrs: list[Union[NumericAttribute, CategoryAttribute]] = []
    for variable in df.columns:
        attribute = series_to_attribute(df[variable])

        attrs.append(attribute)
    blockmodel.attributes = attrs
    blockmodel.validate()

    return blockmodel


