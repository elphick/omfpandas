import tempfile

import numpy as np
import pytest
from pathlib import Path

import omf
from omf import Project

from omfpandas import __omf_version__


def requires_omf_version(version):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if __omf_version__ != version and __omf_version__ not in version.split('|'):
                pytest.skip(f"Test requires omf version {version}, but current version is {__omf_version__}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_omf_file():
    return Path(__file__).resolve().parents[1] / f'assets/{__omf_version__}/test_file.omf'

def get_test_schema():
    return Path(__file__).resolve().parents[1] / f'assets/v2/test_file.schema_with_calc.yaml'

@pytest.fixture
def temp_incongruent_omf_file():
    if __omf_version__ == 'v1':
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{__omf_version__}.omf')
    elif __omf_version__ == 'v2':
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
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{__omf_version__}.omf')
        omf.save(project, temp_file.name, mode='w')
    else:
        raise NotImplementedError(f"Version {__omf_version__} not implemented")

    return Path(temp_file.name)


@pytest.fixture
def temp_congruent_omf_file():
    if __omf_version__ == 'v1':
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{__omf_version__}.omf')
    elif __omf_version__ == 'v2':
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
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{__omf_version__}.omf')
        omf.save(project, temp_file.name, mode='w')
    else:
        raise NotImplementedError(f"Version {__omf_version__} not implemented")
    return Path(temp_file.name)
