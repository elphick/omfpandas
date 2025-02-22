from typing import Optional, Union

import pandas as pd
from omf import TensorGridBlockModel, RegularBlockModel, NumericAttribute, CategoryAttribute

from omfpandas.blockmodels.attributes import read_blockmodel_attributes, BM, series_to_attribute
from omfpandas.blockmodels.geometry import RegularGeometry, TensorGeometry


def df_to_blockmodel(df: pd.DataFrame, blockmodel_name: str) -> Union[RegularBlockModel, TensorGridBlockModel]:
    """
    Get the appropriate function to convert a DataFrame to a BlockModel.

    Args:
        df (pd.DataFrame): The DataFrame to convert to a RegularBlockModel.
        blockmodel_name (str): The name of the RegularBlockModel.

    Returns:
        The RegularBlockModel|TensorGridBlockModel representing the DataFrame.
    """

    if 'x' not in df.index.names and 'y' not in df.index.names and 'z' not in df.index.names:
        raise ValueError("Dataframe must have centroid coordinates (x, y, z) in the index.")
    elif 'dx' in df.index.names and 'dy' in df.index.names and 'dz' in df.index.names:
        return df_to_tensor_bm(df=df, blockmodel_name=blockmodel_name)
    else:
        return df_to_regular_bm(df=df, blockmodel_name=blockmodel_name)


def blockmodel_to_df(blockmodel: BM,
                     variables: Optional[list[str]] = None,
                     query: Optional[str] = None,
                     index_filter: Optional[list[int]] = None) -> pd.DataFrame:
    """Convert regular block model to a DataFrame.

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


def df_to_regular_bm(df: pd.DataFrame, blockmodel_name: str) -> RegularBlockModel:
    """Convert a DataFrame to a RegularBlockModel.

    Args:
        df (pd.DataFrame): The DataFrame to convert to a RegularBlockModel.
        blockmodel_name (str): The name of the RegularBlockModel.

    Returns:
        RegularBlockModel: The RegularBlockModel representing the DataFrame.
    """

    # Sort the dataframe to align with the omf spec
    df.sort_index(level=['z', 'y', 'x'])

    # Create the block model and geometry
    blockmodel = RegularBlockModel(name=blockmodel_name)
    geometry: RegularGeometry = RegularGeometry.from_multi_index(df.index)
    blockmodel.corner = geometry.corner
    blockmodel.axis_u = geometry.axis_u
    blockmodel.axis_v = geometry.axis_v
    blockmodel.axis_w = geometry.axis_w
    blockmodel.block_count = list(geometry.shape)
    blockmodel.block_size = list(geometry.block_size)
    blockmodel.cbc = [1] * geometry.num_cells

    # add the data
    attrs: list[Union[NumericAttribute, CategoryAttribute]] = []
    for variable in df.columns:
        attribute = series_to_attribute(df[variable])

        attrs.append(attribute)
    blockmodel.attributes = attrs
    blockmodel.validate()

    return blockmodel


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
