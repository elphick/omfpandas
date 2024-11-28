import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TypeVar, Union

import numpy as np
import pandas as pd
from omf import NumericAttribute, CategoryAttribute, CategoryColormap
from omf.blockmodel import BaseBlockModel, RegularBlockModel, TensorGridBlockModel
from pandas.core.dtypes.common import is_integer_dtype

from omfpandas.utils.pandas import is_nullable_integer_dtype, to_nullable_integer_dtype, to_numpy_integer_dtype, \
    parse_vars_from_expr

# generic type variable, used for type hinting, to indicate that the type is a subclass of BaseBlockModel
BM = TypeVar('BM', bound=BaseBlockModel)

SENTINEL_VALUE = -9  # TODO: possibly move to config file


@dataclass
class TensorGeometry:
    corner: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    axis_w: np.ndarray
    tensor_u: np.ndarray
    tensor_v: np.ndarray
    tensor_w: np.ndarray
    _centroid_u: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _centroid_v: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _centroid_w: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    @property
    def centroid_u(self) -> np.ndarray:
        if self._centroid_u is None:
            self._centroid_u = (self.tensor_u[:-1] + self.tensor_u[1:]) / 2
        return self._centroid_u

    @property
    def centroid_v(self) -> np.ndarray:
        if self._centroid_v is None:
            self._centroid_v = (self.tensor_v[:-1] + self.tensor_v[1:]) / 2
        return self._centroid_v

    @property
    def centroid_w(self) -> np.ndarray:
        if self._centroid_w is None:
            self._centroid_w = (self.tensor_w[:-1] + self.tensor_w[1:]) / 2
        return self._centroid_w

    def is_regular(self) -> bool:
        return (np.allclose(self.tensor_u, self.tensor_u[0]) and
                np.allclose(self.tensor_v, self.tensor_v[0]) and
                np.allclose(self.tensor_w, self.tensor_w[0]))

    def num_cells(self) -> int:
        return self.tensor_u.size * self.tensor_v.size * self.tensor_w.size

    def shape(self) -> tuple[int, int, int]:
        return self.tensor_u.size, self.tensor_v.size, self.tensor_w.size

    def cell_sizes(self) -> list[tuple[float, float, float]]:
        return list(set(zip(self.tensor_u, self.tensor_v, self.tensor_w)))

    def extents(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        return ((float(self.centroid_u[0] - self.tensor_u[0] / 2), float(self.centroid_u[-1] + self.tensor_u[0] / 2)),
                (float(self.centroid_v[0] - self.tensor_v[0] / 2), float(self.centroid_v[-1] + self.tensor_v[0] / 2)),
                (float(self.centroid_w[0] - self.tensor_w[0] / 2), float(self.centroid_w[-1] + self.tensor_w[0] / 2)))

    def bounding_box(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((float(self.centroid_u[0] - self.tensor_u[0] / 2), float(self.centroid_u[-1] + self.tensor_u[0] / 2)),
                (float(self.centroid_v[0] - self.tensor_v[0] / 2), float(self.centroid_v[-1] + self.tensor_v[0] / 2)))

    def to_dict(self):
        return {
            'corner': self.corner.tolist(),
            'axis_u': self.axis_u.tolist(),
            'axis_v': self.axis_v.tolist(),
            'axis_w': self.axis_w.tolist(),
            'is_regular': self.is_regular(),
            'num_cells': self.num_cells(),
            'shape': self.shape(),
            'cell_sizes': self.cell_sizes(),
            'extents': self.extents(),
            'bounding_box': self.bounding_box()
        }

    def __repr__(self):
        return f"TensorGeometry(corner={self.corner}, axis_u={self.axis_u}, axis_v={self.axis_v}, axis_w={self.axis_w}, " \
               f"is_regular={self.is_regular()}, num_cells={self.num_cells()}, shape={self.shape()}, " \
               f"cell_sizes={self.cell_sizes()}, extents={self.extents()}, bounding_box={self.bounding_box()})"

    def __str__(self):
        return f"TensorGeometry(corner={self.corner}, axis_u={self.axis_u}, axis_v={self.axis_v}, axis_w={self.axis_w}, " \
               f"is_regular={self.is_regular()}, num_cells={self.num_cells()}, shape={self.shape()}, " \
               f"cell_sizes={self.cell_sizes()}, extents={self.extents()}, bounding_box={self.bounding_box()})"

    def to_json(self):
        import json
        return json.dumps(self.to_dict())

    def to_json_file(self, file_path: Path):
        import json
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f)

    def to_yaml(self):
        import yaml
        return yaml.dump(self.to_dict())

    def to_yaml_file(self, file_path: Path):
        import yaml
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f)


def blockmodel_to_df(blockmodel: BM, variables: Optional[list[str]] = None,
                     query: Optional[str] = None,
                     index_filter: Optional[list[int]] = None) -> pd.DataFrame:
    """Convert block model to a DataFrame.

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
            blockmodel.corner = geometry.corner
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
            blockmodel.corner = geometry.corner
            blockmodel.axis_u = geometry.axis_u
            blockmodel.axis_v = geometry.axis_v
            blockmodel.axis_w = geometry.axis_w
            blockmodel.block_count = np.ndarray(
                [geometry.tensor_u.size, geometry.tensor_v.size, geometry.tensor_w.size])
            blockmodel.block_size = np.ndarray([geometry.tensor_u[0], geometry.tensor_v[0], geometry.tensor_w[0]])

        # add the data
        attrs: list[Union[NumericAttribute, CategoryAttribute]] = []
        for variable in df.columns:
            attribute = series_to_attribute(df[variable])

            attrs.append(attribute)
        blockmodel.attributes = attrs

    finally:
        # Reset the index to the original order (to avoid side effects)
        df = df.reindex(original_order)

    return blockmodel


def series_to_attribute(series: pd.Series) -> Union[CategoryAttribute, NumericAttribute]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        cat_map = {i: c for i, c in enumerate(series.cat.categories)}
        cat_col_map = CategoryColormap(indices=list(cat_map.keys()), values=list(cat_map.values()))
        attribute = CategoryAttribute(name=series.name, location="cells", array=np.array(series.cat.codes),
                                      categories=cat_col_map)
    else:
        # manage the sentinel / null placeholders
        # REF: https://github.com/gmggroup/omf-python/issues/59
        if is_nullable_integer_dtype(series):
            # set null_values and assign metadata
            data: pd.Series = series.fillna(SENTINEL_VALUE).pipe(to_numpy_integer_dtype)
            attribute = NumericAttribute(name=series.name, location="cells", array=data.values)
            attribute.metadata['null_value'] = SENTINEL_VALUE
        elif is_integer_dtype(series):
            attribute = NumericAttribute(name=series.name, location="cells", array=series.values)
            attribute.metadata['null_value'] = SENTINEL_VALUE
        else:
            attribute = NumericAttribute(name=series.name, location="cells", array=series.values)
            attribute.metadata['null_value'] = 'np.nan'
    return attribute


def attribute_to_series(attribute: Union[CategoryAttribute, NumericAttribute]) -> pd.Series:
    if isinstance(attribute, CategoryAttribute):
        return pd.Series(pd.Categorical.from_codes(codes=attribute.array.array.ravel(),
                                                   categories=attribute.categories.values,
                                                   ordered=False), name=attribute.name)
    else:
        # if an int with null_value in metadata then convert to a nullable int
        if attribute.metadata.get("null_value") and is_integer_dtype(attribute.array.array):
            return pd.Series(attribute.array.array.ravel(), name=attribute.name).pipe(
                to_nullable_integer_dtype).replace(
                SENTINEL_VALUE, pd.NA)
        return pd.Series(attribute.array.array.ravel(), name=attribute.name, dtype=attribute.array.array.dtype)


def blockmodel_to_parquet(blockmodel: BM, out_path: Optional[Path] = None,
                          variables: Optional[list[str]] = None,
                          allow_overwrite: bool = False):
    """Convert blockmodel to a Parquet file.

    Args:
        blockmodel (BlockModel): The BlockModel to convert.
        out_path (Optional[Path]): The path to the Parquet file to write. If None, a file with the blockmodel name is
        created.
        variables (Optional[list[str]]): The variables to include in the DataFrame. If None, all variables are included.
        allow_overwrite (bool): If True, overwrite the existing Parquet file. Default is False.

    Raises:
        FileExistsError: If the file already exists and allow_overwrite is False.
    """
    if out_path is None:
        out_path = Path(f"{blockmodel.name}.parquet")
    if out_path.exists() and not allow_overwrite:
        raise FileExistsError(f"File already exists: {out_path}. If you want to overwrite, set allow_overwrite=True.")
    df: pd.DataFrame = blockmodel_to_df(blockmodel, variables=variables)
    df.to_parquet(out_path)


def read_blockmodel_attributes(blockmodel: BM, attributes: Optional[list[str]] = None,
                               query: Optional[str] = None, index_filter: Optional[list[int]] = None) -> pd.DataFrame:
    """Read the attributes/variables from the BlockModel, including calculated attributes.

    Args:
        blockmodel (BlockModel): The BlockModel to read from.
        attributes (list[str]): The attributes to include in the DataFrame.
        query (str): The query to filter the DataFrame.
        index_filter (list[int]): List of integer indices to filter the DataFrame.

    Returns:
        pd.DataFrame: The DataFrame representing the attributes in the BlockModel.

    Raises:
        ValueError: If the attribute is not found in the BlockModel or if both query and index_filter are provided.
    """
    if query and index_filter:
        raise ValueError("Cannot use both query and index_filter at the same time.")

    # identify 'cell' variables in the file
    attributes_available = [v.name for v in blockmodel.attributes if v.location == 'cells']

    # Retrieve calculated attributes from metadata
    calculated_attributes: dict[str, str] = blockmodel.metadata.get('calculated_attributes', {})

    attributes: list[str] = attributes or (attributes_available + list(calculated_attributes.keys()))

    # check if the variables are available
    if not set(attributes).issubset(attributes_available + list(calculated_attributes.keys())):
        raise ValueError(
            f"Variables {set(attributes).difference(attributes_available + list(calculated_attributes.keys()))} "
            f"not found in the BlockModel.")

    int_index: np.ndarray = np.arange(blockmodel.num_cells)
    if query is not None:
        # parse out the attributes from the query using a package
        query_attrs = parse_vars_from_expr(query)
        # check if the attributes in the query are available
        if not set(query_attrs).issubset(attributes_available + list(calculated_attributes.keys())):
            raise ValueError(
                f"Variables {set(query_attrs).difference(attributes_available + list(calculated_attributes.keys()))} "
                f"not found in the BlockModel.")
        query_series: list = []
        for attr_name in query_attrs:
            if attr_name in calculated_attributes:
                query_series.append(
                    _evaluate_calculated_attribute(blockmodel, attr_name, calculated_attributes[attr_name],
                                                   attributes_available))
            else:
                query_series.append(attribute_to_series(_get_attribute_by_name(blockmodel, attr_name)))
        df_to_query: pd.DataFrame = pd.concat(query_series, axis=1)
        int_index = np.array(df_to_query.query(query).index)
    elif index_filter is not None:
        int_index = np.array(index_filter)

    # Loop over the variables
    chunks: list = []
    attr: str
    for attr in attributes:
        if attr in calculated_attributes:
            # Evaluate the calculated attribute
            calculated_series = _evaluate_calculated_attribute(blockmodel, attr, calculated_attributes[attr],
                                                               attributes_available)
            chunks.append(calculated_series.iloc[int_index])
        else:
            attr: Union[CategoryAttribute, NumericAttribute] = _get_attribute_by_name(blockmodel, attr)
            chunks.append(attribute_to_series(attr).iloc[int_index])

    # create the geometry index
    geometry_index = create_index(blockmodel)
    if (query is not None) or (index_filter is not None):
        # filter the index to match the int_index positional index
        geometry_index = geometry_index.take(int_index)

    res = pd.concat(chunks, axis=1)
    res.index = geometry_index
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
    """Convert a MultiIndex to a TensorGridGeometry.

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
    geometry: TensorGeometry = TensorGeometry(corner=origin, axis_u=axis_u, axis_v=axis_v, axis_w=axis_w,
                                              tensor_u=tensor_u, tensor_v=tensor_v, tensor_w=tensor_w)

    return geometry


def _get_attribute_by_name(blockmodel: BM, attr_name: str) -> Union[CategoryAttribute, NumericAttribute]:
    """Get the variable/attribute by its name from a BlockModel.

    Args:
        blockmodel (BlockModel): The BlockModel to get the data from.
        attr_name (str): The name of the attribute to retrieve.

    Returns:
        Union[CategoryAttribute, NumericAttribute]: The attribute with the given name.

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


def _evaluate_calculated_attribute(blockmodel: BM, attr_name: str, calculated_expression: str,
                                   attributes_available: list[str]) -> pd.Series:
    """Evaluate a calculated attribute using the blockmodel and available attributes.

    Args:
        blockmodel (BlockModel): The BlockModel to read from.
        attr_name (str): The name of the calculated attribute.
        calculated_expression (str): The expression to evaluate.
        attributes_available (list[str]): List of available attributes in the BlockModel.

    Returns:
        pd.Series: The evaluated calculated attribute as a pandas Series.
    """
    local_dict = {attr: attribute_to_series(_get_attribute_by_name(blockmodel, attr)) for attr in attributes_available}
    return pd.Series(eval(calculated_expression, {}, local_dict), name=attr_name)
