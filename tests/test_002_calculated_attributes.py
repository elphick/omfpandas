import tempfile
from pathlib import Path

import numpy as np
import omf
import pandas as pd
from omf import NumericAttribute, TensorGridBlockModel, Project

from omfpandas import OMFPandasWriter


def create_test_omf_file() -> Path:
    project = Project(name='Test Project')

    block_model = TensorGridBlockModel(
        name='BlockModel1',
        tensor_u=np.full(10, 10, dtype='float32'),
        tensor_v=np.full(10, 10, dtype='float32'),
        tensor_w=np.full(10, 10, dtype='float32'),
        attributes=[
            NumericAttribute(name='attr1', location="cells", array=np.random.rand(10 * 10 * 10).ravel()),
            NumericAttribute(name='attr2', location="cells", array=np.random.rand(10 * 10 * 10).ravel()),
        ]
    )

    project.elements = [block_model]

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.omf')
    omf.save(project, temp_file.name, mode='w')
    return Path(temp_file.name)


def test_add_calculated_blockmodel_attributes():
    temp_omf_path = create_test_omf_file()
    writer = OMFPandasWriter(temp_omf_path)

    calc_definitions = {
        'calc_attr1': 'attr1 + attr2',
        'calc_attr2': 'attr1 * 2',
        'calc_attr3': 'attr1'
    }

    writer.add_calculated_blockmodel_attributes('BlockModel1', calc_definitions)

    bm = writer.get_element_by_name('BlockModel1')
    assert 'calc_attr1' in bm.metadata['calculated_attributes']
    assert 'calc_attr2' in bm.metadata['calculated_attributes']
    assert bm.metadata['calculated_attributes']['calc_attr1'] == 'attr1 + attr2'
    assert bm.metadata['calculated_attributes']['calc_attr2'] == 'attr1 * 2'

    df: pd.DataFrame = writer.read_blockmodel('BlockModel1', attributes=['attr1', 'calc_attr3'])
    # assert the two columns are equal
    assert np.allclose(df['attr1'], df['calc_attr3'])

    temp_omf_path.unlink()
