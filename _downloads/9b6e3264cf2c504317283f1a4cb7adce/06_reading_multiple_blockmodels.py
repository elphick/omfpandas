"""
Reading Multiple Block Models
=============================

OMF allows storage of multiple block models in a single file.  This example demonstrates how to read multiple
block models to return a single pandas dataframe
"""

import tempfile
from pathlib import Path

import numpy as np
import omf

from omfpandas import OMFPandasReader


# %%
# Instantiate
# -----------
# Create a function to generate a temporary omf file containing two block models.

def create_demo_tensor_model(shape: tuple[int, int, int],
                             cell_size: tuple[float, float, float],
                             name: str,
                             attr_names: list[str]) -> omf.TensorGridBlockModel:
    """Create a tensor grid block model with random attributes"""
    return omf.TensorGridBlockModel(
        name=name,
        tensor_u=np.full(shape[0], cell_size[0], dtype='float32'),
        tensor_v=np.full(shape[1], cell_size[1], dtype='float32'),
        tensor_w=np.full(shape[2], cell_size[2], dtype='float32'),
        attributes=[omf.NumericAttribute(name=attr_name, location="cells",
                                         array=np.random.rand(shape[0] * shape[1] * shape[2]).ravel()) for
                    attr_name in attr_names])


def create_temp_omf_file() -> Path:
    """Creates an omf file directly using omf and numpy"""

    project = omf.Project(name='Test Project')

    # Create block models
    block_model_1 = create_demo_tensor_model((10, 10, 10), (10, 10, 10), 'BlockModel1', ['attr1', 'attr2'])

    block_model_2 = create_demo_tensor_model((10, 10, 10), (10, 10, 10), 'BlockModel2', ['attr3', 'attr4'])

    project.elements = [block_model_1, block_model_2]

    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.omf')
    omf.save(project, temp_file.name, mode='w')
    return Path(temp_file.name)


temp_omf_path = create_temp_omf_file()
reader = OMFPandasReader(temp_omf_path)
reader

# %%
# Read from block models
# ----------------------
# We'll demonstrate how to selectively merge attributes from each of the two block models to create a single dataframe.
# The `read_block_models` method accepts a dictionary where the keys are the block model names and the values are lists
# of attribute names to include in the dataframe.  If None is provided for the attribute list, all attributes
#

blockmodel_attributes = {'BlockModel1': None, 'BlockModel2': ['attr4']}

df = reader.read_block_models(blockmodel_attributes)
df.head(20)

# %%
# Check if the DataFrame contains the expected columns

expected_columns = ['attr1', 'attr2', 'attr4']
assert all(col in df.columns for col in expected_columns)

# %%
# Clean up temporary file

temp_omf_path.unlink()
