from typing import Union, TypeVar, Optional

import numpy as np
import pandas as pd
from omf import CategoryAttribute, NumericAttribute, CategoryColormap
from omf.blockmodel import BaseBlockModel, RegularBlockModel, TensorGridBlockModel
from pandas.core.dtypes.common import is_integer_dtype

from omfpandas.blockmodels.geometry import RegularGeometry, TensorGeometry
from omfpandas.utils.pandas_utils import is_nullable_integer_dtype, to_numpy_integer_dtype, to_nullable_integer_dtype, \
    parse_vars_from_expr

# generic type variable, used for type hinting, to indicate that the type is a subclass of BaseBlockModel
BM = TypeVar('BM', bound=BaseBlockModel)

SENTINEL_VALUE = -9  # TODO: possibly move to config file


def series_to_attribute(series: pd.Series) -> Union[CategoryAttribute, NumericAttribute]:

    # todo manage sorting - see attribute_to_series
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


def get_attribute_by_name(blockmodel: BM, attr_name: str) -> Union[CategoryAttribute, NumericAttribute]:
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


def evaluate_calculated_attribute(blockmodel: BM, attr_name: str, calculated_expression: str,
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
    local_dict = {attr: attribute_to_series(get_attribute_by_name(blockmodel, attr)) for attr in attributes_available}
    return pd.Series(eval(calculated_expression, {}, local_dict), name=attr_name)


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
                    evaluate_calculated_attribute(blockmodel, attr_name, calculated_attributes[attr_name],
                                                  attributes_available))
            else:
                query_series.append(attribute_to_series(get_attribute_by_name(blockmodel, attr_name)))
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
            calculated_series = evaluate_calculated_attribute(blockmodel, attr, calculated_attributes[attr],
                                                              attributes_available)
            chunks.append(calculated_series.iloc[int_index])
        else:
            attr: Union[CategoryAttribute, NumericAttribute] = get_attribute_by_name(blockmodel, attr)
            chunks.append(attribute_to_series(attr).iloc[int_index])

    # create the geometry index
    if isinstance(blockmodel, RegularBlockModel):
        geometry_index: pd.MultiIndex = RegularGeometry.from_element(blockmodel).to_multi_index()
    elif isinstance(blockmodel, TensorGridBlockModel):
        geometry_index: pd.MultiIndex = TensorGeometry.from_element(blockmodel).to_multi_index()
    else:
        raise ValueError(f"BlockModel type {blockmodel.__class__.__name__} not (yet) supported.")

    if (query is not None) or (index_filter is not None):
        # filter the index to match the int_index positional index
        geometry_index = geometry_index.take(int_index)

    res = pd.concat(chunks, axis=1)
    # res.index = geometry_index.to_frame().reset_index(drop=True).sort_values(by=['z', 'y', 'x']).set_index(geometry_index.names).index
    res.index = geometry_index
    return res if isinstance(res, pd.DataFrame) else res.to_frame()
