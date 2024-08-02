from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeVar, Union

import numpy as np
import pandas as pd
from omf import NumericAttribute, CategoryAttribute, CategoryColormap, Array
from omf.base import ProjectElementAttribute

from omf.blockmodel import BaseBlockModel, RegularBlockModel, TensorGridBlockModel
from pandas.core.dtypes.common import is_integer_dtype

from omfpandas.utils.pandas import is_nullable_integer_dtype, to_nullable_integer_dtype, to_numpy_integer_dtype

# generic type variable, used for type hinting, to indicate that the type is a subclass of BaseBlockModel
BM = TypeVar('BM', bound=BaseBlockModel)

SENTINEL_VALUE = -9  # TODO: possibly move to config file


@dataclass
class TensorGeometry:
    """A dataclass to represent the geometry of a tensor grid block model."""
    origin: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    axis_w: np.ndarray
    tensor_u: np.ndarray
    tensor_v: np.ndarray
    tensor_w: np.ndarray

    def is_regular(self) -> bool:
        """Return True if the tensor grid is regular."""
        return (np.allclose(self.tensor_u, self.tensor_u[0]) and np.allclose(self.tensor_v, self.tensor_v[0]) and
                np.allclose(self.tensor_w, self.tensor_w[0]))


def blockmodel_to_df(blockmodel: BM, variables: Optional[list[str]] = None,
                     with_geometry_index: bool = True) -> pd.DataFrame:
    """Convert block model to a DataFrame.

    Args:
        blockmodel (BlockModel): The BlockModel to convert.
        variables (Optional[list[str]]): The variables to include in the DataFrame. If None, all variables are included.
        with_geometry_index (bool): If True, includes geometry index in the DataFrame. Default is True.

    Returns:
        pd.DataFrame: The DataFrame representing the BlockModel.
    """
    # read the data
    df: pd.DataFrame = read_blockmodel_variables(blockmodel, variables=variables)
    if with_geometry_index:
        df.index = create_index(blockmodel)
    return df


def df_to_blockmodel(df: pd.DataFrame, blockmodel_name: str, is_tensor: bool = True) -> BM:
    """Write a DataFrame to a BlockModel.

    Args:
        df (pd.DataFrame): The DataFrame to convert to a BlockModel.
        blockmodel_name (str): The name of the BlockModel.
        is_tensor (bool): If True, a TensorGridBlockModel will be created. If False, a RegularBlockModel will be
        created.

    Returns:
        BlockModel: The BlockModel representing the DataFrame.
    """
    # Get the original order
    original_order = df.index

    try:
        # Sort the dataframe to align with the omf spec
        df.sort_index(level=['z', 'y', 'x'], inplace=True)

        # Create the blockmodel and geometry
        geometry: TensorGeometry = index_to_geometry(df.index)

        if is_tensor:
            blockmodel: BM = TensorGridBlockModel(name=blockmodel_name)
            # assign the geometry properties
            blockmodel.corner = geometry.origin
            blockmodel.axis_u = geometry.axis_u
            blockmodel.axis_v = geometry.axis_v
            blockmodel.axis_w = geometry.axis_w
            blockmodel.tensor_u = geometry.tensor_u
            blockmodel.tensor_v = geometry.tensor_v
            blockmodel.tensor_w = geometry.tensor_w
        else:
            if not geometry.is_regular():
                raise ValueError("RegularBlockModel requires a regular grid.")
            blockmodel: BM = RegularBlockModel(name=blockmodel_name)
            blockmodel.corner = geometry.origin
            blockmodel.axis_u = geometry.axis_u
            blockmodel.axis_v = geometry.axis_v
            blockmodel.axis_w = geometry.axis_w
            blockmodel.block_count = np.ndarray(
                [geometry.tensor_u.size, geometry.tensor_v.size, geometry.tensor_w.size])
            blockmodel.block_size = np.ndarray([geometry.tensor_u[0], geometry.tensor_v[0], geometry.tensor_w[0]])

        # add the data
        attrs: list[Union[NumericAttribute, CategoryAttribute]] = []
        for variable in df.columns:
            if isinstance(df[variable].dtype, pd.CategoricalDtype):
                cat_map = {i: c for i, c in enumerate(df[variable].cat.categories)}
                cat_col_map = CategoryColormap(indices=list(cat_map.keys()), values=list(cat_map.values()))
                attribute = CategoryAttribute(name=variable, location="cells", array=np.array(df[variable].cat.codes),
                                              categories=cat_col_map)
            else:
                # manage the sentinel / null placeholders
                # REF: https://github.com/gmggroup/omf-python/issues/59
                if is_nullable_integer_dtype(df[variable]):
                    # set null_values and assign metadata
                    data: pd.Series = df[variable].fillna(SENTINEL_VALUE).pipe(to_numpy_integer_dtype)
                    attribute = NumericAttribute(name=variable, location="cells", array=data.values)
                    attribute.metadata['null_value'] = SENTINEL_VALUE
                elif is_integer_dtype(df[variable]):
                    attribute = NumericAttribute(name=variable, location="cells", array=df[variable].values)
                    attribute.metadata['null_value'] = SENTINEL_VALUE
                else:
                    attribute = NumericAttribute(name=variable, location="cells", array=df[variable].values)
                    attribute.metadata['null_value'] = 'np.nan'

            attrs.append(attribute)
        blockmodel.attributes = attrs

    finally:
        # Reset the index to the original order (to avoid side effects)
        df = df.reindex(original_order)

    return blockmodel


def blockmodel_to_parquet(blockmodel: BM, out_path: Optional[Path] = None,
                          variables: Optional[list[str]] = None,
                          with_geometry_index: bool = True, allow_overwrite: bool = False):
    """Convert blockmodel to a Parquet file.

    Args:
        blockmodel (BlockModel): The BlockModel to convert.
        out_path (Optional[Path]): The path to the Parquet file to write. If None, a file with the blockmodel name is
        created.
        variables (Optional[list[str]]): The variables to include in the DataFrame. If None, all variables are included.
        with_geometry_index (bool): If True, includes geometry index in the DataFrame. Default is True.
        allow_overwrite (bool): If True, overwrite the existing Parquet file. Default is False.

    Raises:
        FileExistsError: If the file already exists and allow_overwrite is False.
    """
    if out_path is None:
        out_path = Path(f"{blockmodel.name}.parquet")
    if out_path.exists() and not allow_overwrite:
        raise FileExistsError(f"File already exists: {out_path}. If you want to overwrite, set allow_overwrite=True.")
    df: pd.DataFrame = blockmodel_to_df(blockmodel, variables=variables, with_geometry_index=with_geometry_index)
    df.to_parquet(out_path)


def read_blockmodel_variables(blockmodel: BM, variables: Optional[list[str]] = None) -> pd.DataFrame:
    """Read the variables from the BlockModel.

    Args:
        blockmodel (BlockModel): The BlockModel to read from.
        variables (list[str]): The variables to include in the DataFrame.

    Returns:
        pd.DataFrame: The DataFrame representing the variables in the BlockModel.

    Raises:
        ValueError: If the variable is not found in the BlockModel.
    """

    # identify 'cell' variables in the file
    variables_available = [v.name for v in blockmodel.attributes if v.location == 'cells']

    variables = variables or variables_available

    # check if the variables are available
    if not set(variables).issubset(variables_available):
        raise ValueError(f"Variables {set(variables).difference(variables_available)} not found in the BlockModel.")

    # Loop over the variables
    chunks: list = []
    for variable in variables:
        attr: ProjectElementAttribute = _get_attribute_by_name(blockmodel, variable)
        if isinstance(attr, CategoryAttribute):
            attr_series: pd.Series = pd.Series(pd.Categorical.from_codes(codes=attr.array.array.ravel(),
                                                                         categories=attr.categories.values,
                                                                         ordered=False), name=variable)
        else:
            attr_series: pd.Series = pd.Series(attr.array.array.ravel(), name=variable, dtype=attr.array.array.dtype)
            # if an int with null_value in metadata then convert to a nullable int
            if attr.metadata.get("null_value") and is_integer_dtype(attr_series):
                attr_series = attr_series.pipe(to_nullable_integer_dtype).replace(SENTINEL_VALUE, pd.NA).astype(
                    str(attr_series.dtype).replace("i", "I"))

        chunks.append(attr_series)

    res = pd.concat(chunks, axis=1)
    return res if isinstance(res, pd.DataFrame) else res.to_frame()


def create_index(blockmodel: BM) -> pd.MultiIndex:
    """Returns a pd.MultiIndex for the blockmodel element.

    Args:
        blockmodel (BlockModel): The BlockModel to get the index from.

    Returns:
        pd.MultiIndex: The MultiIndex representing the blockmodel element geometry.
    """
    ox, oy, oz = blockmodel.corner

    # Make coordinates (points) along each axis, i, j, k
    i = ox + np.cumsum(blockmodel.tensor_u)
    i = np.insert(i, 0, ox)
    j = oy + np.cumsum(blockmodel.tensor_v)
    j = np.insert(j, 0, oy)
    k = oz + np.cumsum(blockmodel.tensor_w)
    k = np.insert(k, 0, oz)

    # convert to centroids
    x, y, z = (i[1:] + i[:-1]) / 2, (j[1:] + j[:-1]) / 2, (k[1:] + k[:-1]) / 2
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    # Calculate dx, dy, dz
    dxx, dyy, dzz = np.meshgrid(blockmodel.tensor_u, blockmodel.tensor_v, blockmodel.tensor_w, indexing="ij")

    # TODO: consider rotation

    index = pd.MultiIndex.from_arrays([xx.ravel("F"), yy.ravel("F"), zz.ravel("F"),
                                       dxx.ravel("F"), dyy.ravel("F"), dzz.ravel("F")],
                                      names=['x', 'y', 'z', 'dx', 'dy', 'dz'])

    return index


def index_to_geometry(index: pd.MultiIndex) -> TensorGeometry:
    """Convert a MultiIndex to a VolumeGridGeometry.

    Args:
        index (pd.MultiIndex): The MultiIndex to convert to a TensorGeometry.

    Returns:
        TensorGeometry: The TensorGeometry representing the MultiIndex.
    """
    # check that the index contains the expected levels
    if not {'x', 'y', 'z', 'dx', 'dy', 'dz'}.issubset(index.names):
        raise ValueError("Index must contain the levels 'x', 'y', 'z', 'dx', 'dy', 'dz'.")

    x = index.get_level_values('x').unique()
    y = index.get_level_values('y').unique()
    z = index.get_level_values('z').unique()

    # Get the shape of the original 3D arrays
    shape = (len(x), len(y), len(z))

    # Reshape the ravelled index back into the original shapes
    tensor_u = index.get_level_values('dx').values.reshape(shape, order='F')[:, 0, 0]
    tensor_v = index.get_level_values('dy').values.reshape(shape, order='F')[0, :, 0]
    tensor_w = index.get_level_values('dz').values.reshape(shape, order='F')[0, 0, :]

    origin_x = x.min() - tensor_u[0] / 2
    origin_y = y.min() - tensor_v[0] / 2
    origin_z = z.min() - tensor_w[0] / 2

    # Create the geometry
    origin = np.array([origin_x, origin_y, origin_z])
    axis_u = np.array([1, 0, 0])
    axis_v = np.array([0, 1, 0])
    axis_w = np.array([0, 0, 1])
    geometry: TensorGeometry = TensorGeometry(origin=origin, axis_u=axis_u, axis_v=axis_v, axis_w=axis_w,
                                              tensor_u=tensor_u, tensor_v=tensor_v, tensor_w=tensor_w)

    return geometry


def _get_attribute_by_name(blockmodel: BM, attr_name: str) -> ProjectElementAttribute:
    """Get the variable/attribute by its name from a BlockModel.

    Args:
        blockmodel (BlockModel): The BlockModel to get the data from.
        attr_name (str): The name of the attribute to retrieve.

    Returns:
        ProjectElementAttribute: The attribute with the given name.

    Raises:
        ValueError: If the variable is not found as cell data in the BlockModel or if multiple variables with the
        same name are found.
    """
    attrs = [sd for sd in blockmodel.attributes if sd.location == 'cells' and sd.name == attr_name]
    if not attrs:
        raise ValueError(f"Variable '{attr_name}' not found as cell data in the BlockModel: {blockmodel}")
    elif len(attrs) > 1:
        raise ValueError(f"Multiple variables with the name '{attr_name}' found in the BlockModel: {blockmodel}")
    return attrs[0]
