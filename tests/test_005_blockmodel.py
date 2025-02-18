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


def test_incorrect_file_version():
    correct_path: Path = get_omf_file()
    if __omf_version__ == 'v1':
        test_omf_path: Path = Path(str(correct_path.parent).replace('v1', 'v2')) / correct_path.name
    elif __omf_version__ == 'v2':
        test_omf_path: Path = Path(str(correct_path.parent).replace('v2', 'v1')) / correct_path.name
    else:
        raise ValueError('Unsupported OMF version')

    with pytest.raises(ValueError, match='Invalid OMF file'):
        omfp: OMFPandasReader = OMFPandasReader(filepath=test_omf_path)


def test_blockmodel_to_dataframe():
    # Test the OMFBlockModel class
    test_omf_path: Path = get_omf_file()  #
    omfp: OMFPandasReader = OMFPandasReader(filepath=test_omf_path)

    if __omf_version__ == 'v1':
        bm_element_name: str = 'Block Model'
    elif __omf_version__ == 'v2':
        bm_element_name: str = 'tensor'
    else:
        raise ValueError('Unsupported OMF version')

    bm_el = omfp.get_element_by_name(bm_element_name)

    assert bm_el.name == bm_element_name

    bm: OMFBlockModel = OMFBlockModel(bm_el)
    df: pd.DataFrame = bm.to_dataframe()
    assert df is not None


def test_blockmodel_from_dataframe():
    # Test the OMFBlockModel class
    test_omf_path: Path = get_omf_file()  #
    omfp: OMFPandasReader = OMFPandasReader(filepath=test_omf_path)

    if __omf_version__ == 'v1':
        bm_element_name: str = 'Block Model'
        blockmodel_type: Literal['regular'] = 'regular'
        expected_bm_type: str = 'VolumeElement'
    elif __omf_version__ == 'v2':
        bm_element_name: str = 'regular'
        blockmodel_type: Literal['regular', 'tensor'] = 'regular'
        expected_bm_type: str = 'RegularBlockModel'
    else:
        raise ValueError('Unsupported OMF version')

    bm_el = omfp.get_element_by_name(bm_element_name)

    assert bm_el.name == bm_element_name

    bm: OMFBlockModel = OMFBlockModel(bm_el)
    df: pd.DataFrame = bm.to_dataframe()

    # create a new block model from the dataframe

    bm2 = OMFBlockModel.from_dataframe(df, blockmodel_name='new_blockmodel', blockmodel_type=blockmodel_type)

    # check the block model name
    assert bm2.blockmodel.name == 'new_blockmodel'
    assert bm2.bm_type == expected_bm_type

    if __omf_version__ == 'v2':
        bm_element_name: str = 'tensor'
        bm_el_tensor = omfp.get_element_by_name(bm_element_name)
        blockmodel_type: Literal['regular', 'tensor'] = 'tensor'
        expected_bm_type: str = 'TensorGridBlockModel'
        df2: pd.DataFrame = OMFBlockModel(bm_el_tensor).to_dataframe()
        bm3 = OMFBlockModel.from_dataframe(df2, blockmodel_name='new_tensor_blockmodel',
                                           blockmodel_type=blockmodel_type)
        assert bm3.blockmodel.name == 'new_tensor_blockmodel'
        assert bm3.bm_type == expected_bm_type

        # test with No blockmodel_type specified
        bm4 = OMFBlockModel.from_dataframe(df2, blockmodel_name='new_tensor_blockmodel', blockmodel_type=None)
        assert bm4.blockmodel.name == 'new_tensor_blockmodel'
        assert bm4.bm_type == expected_bm_type
