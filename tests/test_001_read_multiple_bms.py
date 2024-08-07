import tempfile

import numpy as np
import omf
import pandas as pd
from pathlib import Path

import pytest
from omf import Project, NumericAttribute, TensorGridBlockModel

from omfpandas.reader import OMFPandasReader


def create_incongruent_omf_file() -> Path:
    project = Project(name='Test Project')

    # Create block models
    block_model_1 = TensorGridBlockModel(
        name='BlockModel1',
        tensor_u=np.full(10, 10, dtype='float32'),
        tensor_v=np.full(10, 10, dtype='float32'),
        tensor_w=np.full(10, 10, dtype='float32'),
        attributes=[
            NumericAttribute(name='attr1', location="cells", array=np.random.rand(10 * 10 * 10).ravel()),
            NumericAttribute(name='attr2', location="cells", array=np.random.rand(10 * 10 * 10).ravel()),
        ]
    )

    block_model_2 = TensorGridBlockModel(
        name='BlockModel2',
        tensor_u=np.full(5, 10, dtype='float32'),
        tensor_v=np.full(5, 10, dtype='float32'),
        tensor_w=np.full(5, 10, dtype='float32'),
        attributes=[
            NumericAttribute(name='attr3', location="cells", array=np.random.rand(5 * 5 * 5).ravel()),
            NumericAttribute(name='attr4', location="cells", array=np.random.rand(5 * 5 * 5).ravel()),
        ]
    )

    project.elements = [block_model_1, block_model_2]

    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.omf')
    omf.save(project, temp_file.name, mode='w')
    return Path(temp_file.name)


def create_congruent_omf_file() -> Path:
    project = Project(name='Test Project')

    # Create block models
    block_model_1 = TensorGridBlockModel(
        name='BlockModel1',
        tensor_u=np.full(10, 10, dtype='float32'),
        tensor_v=np.full(10, 10, dtype='float32'),
        tensor_w=np.full(10, 10, dtype='float32'),
        attributes=[
            NumericAttribute(name='attr1', location="cells", array=np.random.rand(10 * 10 * 10).ravel()),
            NumericAttribute(name='attr2', location="cells", array=np.random.rand(10 * 10 * 10).ravel()),
        ]
    )

    block_model_2 = TensorGridBlockModel(
        name='BlockModel2',
        tensor_u=np.full(10, 10, dtype='float32'),
        tensor_v=np.full(10, 10, dtype='float32'),
        tensor_w=np.full(10, 10, dtype='float32'),
        attributes=[
            NumericAttribute(name='attr3', location="cells", array=np.random.rand(10 * 10 * 10).ravel()),
            NumericAttribute(name='attr4', location="cells", array=np.random.rand(10 * 10 * 10).ravel()),
        ]
    )

    project.elements = [block_model_1, block_model_2]

    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.omf')
    omf.save(project, temp_file.name, mode='w')
    return Path(temp_file.name)


def test_read_incongruent_block_models():
    temp_omf_path = create_incongruent_omf_file()
    reader = OMFPandasReader(temp_omf_path)

    blockmodel_attributes = {
        'BlockModel1': None,
        'BlockModel2': ['attr4']
    }

    # failure is expected
    with pytest.raises(ValueError):
        df = reader.read_block_models(blockmodel_attributes)

    # Clean up temporary file
    temp_omf_path.unlink()


def test_read_block_models():
    temp_omf_path = create_congruent_omf_file()
    reader = OMFPandasReader(temp_omf_path)

    blockmodel_attributes = {
        'BlockModel1': None,
        'BlockModel2': ['attr4']
    }

    df = reader.read_block_models(blockmodel_attributes)

    # Check if the DataFrame contains the expected columns
    expected_columns = ['attr1', 'attr2', 'attr4']
    assert all(col in df.columns for col in expected_columns)

    # Clean up temporary file
    temp_omf_path.unlink()
