import tempfile
from pathlib import Path

import numpy as np
import omf
import pandas as pd
from omf import Project

from omfpandas import OMFPandasReader
from omfpandas.writer import OMFPandasWriter
from conftest import get_test_schema


def create_test_omf2_file() -> Path:
    from omf import NumericAttribute, TensorGridBlockModel
    project = Project(name='Test Project')

    # create a regular model
    # regular_block_model = RegularBlockModel(...)

    # create a tensor model
    tensor_block_model = TensorGridBlockModel(
        name='TensorModel1',
        tensor_u=np.full(10, 10, dtype='float32'),
        tensor_v=np.full(10, 10, dtype='float32'),
        tensor_w=np.full(10, 10, dtype='float32'),
        attributes=[
            NumericAttribute(name='attr1', location="cells", array=np.random.rand(10 * 10 * 10).ravel()),
            NumericAttribute(name='attr2', location="cells", array=np.random.rand(10 * 10 * 10).ravel()),
        ]
    )

    project.elements = [tensor_block_model]

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.omf')
    omf.save(project, temp_file.name, mode='w')
    return Path(temp_file.name)


def test_add_calculated_blockmodel_attributes():
    temp_omf_path = create_test_omf2_file()
    writer = OMFPandasWriter(temp_omf_path)

    calc_definitions = {
        'calc_attr1': 'attr1 + attr2',
        'calc_attr2': 'attr1 * 2',
        'calc_attr3': 'attr1'
    }

    writer.create_calculated_blockmodel_attributes('TensorModel1', calc_definitions)

    bm = writer.get_element_by_name('TensorModel1')
    assert 'calc_attr1' in bm.metadata['calculated_attributes']
    assert 'calc_attr2' in bm.metadata['calculated_attributes']
    assert 'calc_attr3' in bm.metadata['calculated_attributes']

    assert bm.metadata['calculated_attributes']['calc_attr1'] == 'attr1 + attr2'
    assert bm.metadata['calculated_attributes']['calc_attr2'] == 'attr1 * 2'
    assert bm.metadata['calculated_attributes']['calc_attr3'] == 'attr1'

    df: pd.DataFrame = writer.read_blockmodel('TensorModel1', attributes=['attr1', 'calc_attr3'])
    # assert the two columns are equal
    assert np.allclose(df['attr1'], df['calc_attr3'])

    temp_omf_path.unlink()


def test_add_calculated_blockmodel_attributes_from_schema():
    temp_omf_path = create_test_omf2_file()
    df: pd.DataFrame = OMFPandasReader(temp_omf_path).read_blockmodel('TensorModel1')
    writer = OMFPandasWriter(temp_omf_path)

    writer.create_blockmodel(blocks=df, blockmodel_name='TensorModel1_calc',
                             pd_schema=get_test_schema())

    bm = writer.get_element_by_name('TensorModel1_calc')
    assert 'calc_attr1' in bm.metadata['calculated_attributes']

    assert bm.metadata['calculated_attributes']['calc_attr1'] == 'attr1 * 2'
    assert bm.metadata['calculated_attributes']['calc_attr2'] == 'attr1'

    df: pd.DataFrame = writer.read_blockmodel('TensorModel1_calc', attributes=['attr1', 'calc_attr2'])
    # assert the two columns are equal
    assert np.allclose(df['attr1'], df['calc_attr2'])

    temp_omf_path.unlink()
