import tempfile

import numpy as np
import pytest
from pathlib import Path

import omf
from omf import Project



def get_omf_file():
    return Path(__file__).resolve().parents[1] / f'assets/test_file.omf'

def get_test_schema():
    return Path(__file__).resolve().parents[1] / f'assets/test_file.schema_with_calc.yaml'

@pytest.fixture
def temp_incongruent_omf_file():
    from omf import NumericAttribute, TensorGridBlockModel

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


@pytest.fixture
def temp_congruent_omf_file():
    from omf import NumericAttribute, TensorGridBlockModel
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
