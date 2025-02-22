from pathlib import Path
from typing import Literal

import pandas as pd
import pytest

from omfpandas import OMFPandasReader, __omf_version__
from omfpandas.blockmodel import OMFBlockModel
from conftest import get_omf_file


def test_init():
    # Create the object OMFPandas with the path to the OMF file.
    test_omf_path: Path = get_omf_file()

    #
    omfp: OMFPandasReader = OMFPandasReader(filepath=test_omf_path)
    assert omfp.omf_version == __omf_version__


def test_blockmodel_to_dataframe():
    # Test the OMFBlockModel class
    test_omf_path: Path = get_omf_file()  #
    omfp: OMFPandasReader = OMFPandasReader(filepath=test_omf_path)
    bm_el = omfp.get_element_by_name('tensor')

    bm: OMFBlockModel = OMFBlockModel(blockmodel=bm_el)
    df: pd.DataFrame = bm.to_dataframe()
    assert df is not None


def test_round_trip_regular():
    # Test the OMFBlockModel class
    test_omf_path: Path = get_omf_file()  #
    omfp: OMFPandasReader = OMFPandasReader(filepath=test_omf_path)

    bm = omfp.get_element_by_name('regular')
    assert bm.__class__.__name__ == 'RegularBlockModel'

    bm: OMFBlockModel = OMFBlockModel(blockmodel=bm)
    df: pd.DataFrame = bm.to_dataframe()

    # check the dataframe index has the expected levels
    assert df.index.nlevels == 3
    assert df.index.names == ['x', 'y', 'z']

    # create a new block model from the dataframe

    bm2 = OMFBlockModel.from_dataframe(df, blockmodel_name='new_blockmodel')

    # check the block model name
    assert bm2.blockmodel.name == 'new_blockmodel'
    assert bm2.bm_type == 'RegularBlockModel'


def test_round_trip_tensor():
    # Test the OMFBlockModel class
    test_omf_path: Path = get_omf_file()  #
    omfp: OMFPandasReader = OMFPandasReader(filepath=test_omf_path)

    bm = omfp.get_element_by_name('tensor')
    assert bm.__class__.__name__ == 'TensorGridBlockModel'

    bm: OMFBlockModel = OMFBlockModel(blockmodel=bm)
    df: pd.DataFrame = bm.to_dataframe()

    # check the dataframe index has the expected levels
    assert df.index.nlevels == 6
    assert df.index.names == ['x', 'y', 'z', 'dx', 'dy', 'dz']

    # create a new block model from the dataframe

    bm2 = OMFBlockModel.from_dataframe(df, blockmodel_name='new_blockmodel')

    # check the block model name
    assert bm2.blockmodel.name == 'new_blockmodel'
    assert bm2.bm_type == 'TensorGridBlockModel'
