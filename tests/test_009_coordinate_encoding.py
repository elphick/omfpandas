from pathlib import Path

import pandas as pd
import numpy as np

from omfpandas import OMFPandasReader
from omfpandas.blockmodels import multiindex_to_encoded_index, encoded_index_to_multiindex
from conftest import get_omf_file


def test_coordinate_encoding_round_trip():
    # Create a sample MultiIndex
    x = np.linspace(0, 100, 5)
    y = np.linspace(10000, 10100, 5)
    z = np.linspace(4000, 4200, 5)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    multi_index: pd.MultiIndex = pd.MultiIndex.from_arrays([xx.ravel(), yy.ravel(), zz.ravel()], names=['x', 'y', 'z'])

    # Encode the MultiIndex to an encoded integer Index
    encoded_index: pd.Index = multiindex_to_encoded_index(multi_index)

    # Decode the encoded integer Index back to a MultiIndex
    decoded_multi_index: pd.MultiIndex = encoded_index_to_multiindex(encoded_index)

    # Verify the round-trip conversion
    pd.testing.assert_index_equal(multi_index, decoded_multi_index, check_names=True)


def test_encoded_index_is_monotonic():
    # Create a sample MultiIndex
    x = np.linspace(0, 100, 5)
    y = np.linspace(10000, 10100, 5)
    z = np.linspace(4000, 4200, 5)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    multi_index = pd.MultiIndex.from_arrays([xx.ravel(), yy.ravel(), zz.ravel()], names=['x', 'y', 'z'])

    # Encode the MultiIndex to an encoded integer Index
    encoded_index: pd.Index = multiindex_to_encoded_index(multi_index)

    # Verify that the encoded index is monotonic
    assert encoded_index.is_monotonic_increasing, "Encoded index is not monotonic increasing"


def test_read_blockmodel_encoded_index():
    test_omf_path: Path = get_omf_file()  #
    omfp: OMFPandasReader = OMFPandasReader(filepath=test_omf_path)

    df_multi_index: pd.DataFrame = omfp.read_blockmodel('regular', encode_index=False)
    assert isinstance(df_multi_index.index, pd.MultiIndex)
    assert df_multi_index.index.names == ['x', 'y', 'z']

    df_encoded_index: pd.DataFrame = omfp.read_blockmodel('regular', encode_index=True)
    assert isinstance(df_encoded_index.index, pd.Index)
    assert df_encoded_index.index.name == 'encoded_xyz'
