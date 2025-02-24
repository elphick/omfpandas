"""
OMF uses C-order, sorting by x, y, z, while PyVista uses Fortran-order, sorting by z, y, x.
Our Pandas MultiIndex is sorted by x, y, z to align with OMF.
"""


from typing import Optional, Union

import numpy as np
import pandas as pd
from omf import TensorGridBlockModel, RegularBlockModel, NumericAttribute, CategoryAttribute

from omfpandas.blockmodels.attributes import read_blockmodel_attributes, BM, series_to_attribute
from omfpandas.blockmodels.geometry import RegularGeometry, TensorGeometry

import pyvista as pv

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

    # Sort the dataframe to align with the omf spec - 'C' order
    df.sort_index(level=['x', 'y', 'z'])

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

    # Sort the dataframe to align with the omf spec - 'C' order
    df.sort_index(level=['x', 'y', 'z'], inplace=True)

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


def df_to_pv_structured_grid(df: pd.DataFrame) -> 'pv.StructuredGrid':
    import pyvista as pv

    # ensure the dataframe is sorted by z, y, x, since Pyvista uses 'F' order.
    df = df.sort_index(level=['z', 'y', 'x'])

    # Get the unique x, y, z coordinates (centroids)
    x_centroids = df.index.get_level_values('x').unique()
    y_centroids = df.index.get_level_values('y').unique()
    z_centroids = df.index.get_level_values('z').unique()

    # Calculate the cell size (assuming all cells are of equal size)
    dx = np.diff(x_centroids)[0]
    dy = np.diff(y_centroids)[0]
    dz = np.diff(z_centroids)[0]

    # Calculate the grid points
    x_points = np.concatenate([x_centroids - dx / 2, x_centroids[-1:] + dx / 2])
    y_points = np.concatenate([y_centroids - dy / 2, y_centroids[-1:] + dy / 2])
    z_points = np.concatenate([z_centroids - dz / 2, z_centroids[-1:] + dz / 2])

    # Create the 3D grid of points
    x, y, z = np.meshgrid(x_points, y_points, z_points, indexing='ij')

    # Create a StructuredGrid object
    grid = pv.StructuredGrid(x, y, z)

    # Add the data from the DataFrame to the grid
    for column in df.columns:
        grid.cell_data[column] = df[column].values

    return grid


def df_to_pv_unstructured_grid(df: pd.DataFrame) -> 'pv.UnstructuredGrid':
    """
    Requires the index to be a pd.MultiIndex with names ['x', 'y', 'z', 'dx', 'dy', 'dz'].
    :return:
    """

    # ensure the dataframe is sorted by z, y, x, since Pyvista uses 'F' order.
    blocks = df.reset_index().sort_values(['z', 'y', 'x'])

    # Get the x, y, z coordinates and cell dimensions
    # if no dims are passed, estimate them
    if 'dx' not in blocks.columns:
        dx, dy, dz = df.common_block_size()
        blocks['dx'] = dx
        blocks['dy'] = dy
        blocks['dz'] = dz

    x, y, z, dx, dy, dz = (blocks[col].values for col in blocks.columns if col in ['x', 'y', 'z', 'dx', 'dy', 'dz'])
    blocks.set_index(['x', 'y', 'z', 'dx', 'dy', 'dz'], inplace=True)
    # Create the cell points/vertices
    # REF: https://github.com/OpenGeoVis/PVGeo/blob/main/PVGeo/filters/voxelize.py

    n_cells = len(x)

    # Generate cell nodes for all points in data set
    # - Bottom
    c_n1 = np.stack(((x - dx / 2), (y - dy / 2), (z - dz / 2)), axis=1)
    c_n2 = np.stack(((x + dx / 2), (y - dy / 2), (z - dz / 2)), axis=1)
    c_n3 = np.stack(((x - dx / 2), (y + dy / 2), (z - dz / 2)), axis=1)
    c_n4 = np.stack(((x + dx / 2), (y + dy / 2), (z - dz / 2)), axis=1)
    # - Top
    c_n5 = np.stack(((x - dx / 2), (y - dy / 2), (z + dz / 2)), axis=1)
    c_n6 = np.stack(((x + dx / 2), (y - dy / 2), (z + dz / 2)), axis=1)
    c_n7 = np.stack(((x - dx / 2), (y + dy / 2), (z + dz / 2)), axis=1)
    c_n8 = np.stack(((x + dx / 2), (y + dy / 2), (z + dz / 2)), axis=1)

    # - Concatenate
    # nodes = np.concatenate((c_n1, c_n2, c_n3, c_n4, c_n5, c_n6, c_n7, c_n8), axis=0)
    nodes = np.hstack((c_n1, c_n2, c_n3, c_n4, c_n5, c_n6, c_n7, c_n8)).ravel().reshape(n_cells * 8, 3)

    # create the cells
    # REF: https://docs/pyvista.org/examples/00-load/create-unstructured-surface.html
    cells_hex = np.arange(n_cells * 8).reshape(n_cells, 8)

    grid = pv.UnstructuredGrid({pv.CellType.VOXEL: cells_hex}, nodes)

    # add the attributes (column) data
    for col in blocks.columns:
        grid.cell_data[col] = blocks[col].values

    return grid


def convert_tensor_to_regular(tensor_model: TensorGridBlockModel) -> RegularBlockModel:
    # Check if the TensorGridBlockModel is regularly spaced
    tensor_geometry = TensorGeometry.from_element(tensor_model)
    if not tensor_geometry.is_regular:
        raise ValueError(
            "The TensorGridBlockModel is not regularly spaced and cannot be converted to a RegularBlockModel.")

    # Create a RegularGeometry from the TensorGeometry
    regular_geometry = RegularGeometry(
        corner=tensor_geometry.corner,
        axis_u=tensor_geometry.axis_u,
        axis_v=tensor_geometry.axis_v,
        axis_w=tensor_geometry.axis_w,
        block_size=(tensor_geometry.tensor_u[0], tensor_geometry.tensor_v[0], tensor_geometry.tensor_w[0]),
        shape=tensor_geometry.shape
    )

    # Create a RegularBlockModel using the RegularGeometry
    regular_model = RegularBlockModel(
        origin=list(regular_geometry.corner),
        axis_u=list(regular_geometry.axis_u),
        axis_v=list(regular_geometry.axis_v),
        axis_w=list(regular_geometry.axis_w),
        block_size=list(regular_geometry.block_size),
        block_count=list(regular_geometry.shape),
        cbc = [1] * regular_geometry.num_cells
    )

    # Copy attributes from the TensorGridBlockModel to the RegularBlockModel
    regular_model.attributes = tensor_model.attributes

    return regular_model