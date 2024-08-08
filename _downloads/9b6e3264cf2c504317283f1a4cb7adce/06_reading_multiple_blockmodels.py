"""
Reading Multiple Block Models
=============================

OMF allows storage of multiple block models in a single file.  This example demonstrates how to read multiple
block models to return a single pandas dataframe
"""
import logging
import shutil
import tempfile
import webbrowser
from pathlib import Path

import numpy as np
import omf
import pandas as pd

from omfpandas import OMFDataConverter, OMFPandasReader, OMFPandasWriter

# %%
# Instantiate
# -----------
# Create the object OMFPandas with the path to the OMF file.

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')


# %%
# Create a function to generate a temporary omf file containing two block models.
def create_temp_omf_file() -> Path:
    """Creates an omf file directly using omf and numpy"""

    project = omf.Project(name='Test Project')

    # Create block models
    block_model_1 = omf.TensorGridBlockModel(
        name='BlockModel1',
        tensor_u=np.full(10, 10, dtype='float32'),
        tensor_v=np.full(10, 10, dtype='float32'),
        tensor_w=np.full(10, 10, dtype='float32'),
        attributes=[
            omf.NumericAttribute(name='attr1', location="cells", array=np.random.rand(10 * 10 * 10).ravel()),
            omf.NumericAttribute(name='attr2', location="cells", array=np.random.rand(10 * 10 * 10).ravel()),
        ]
    )

    block_model_2 = omf.TensorGridBlockModel(
        name='BlockModel2',
        tensor_u=np.full(10, 10, dtype='float32'),
        tensor_v=np.full(10, 10, dtype='float32'),
        tensor_w=np.full(10, 10, dtype='float32'),
        attributes=[
            omf.NumericAttribute(name='attr3', location="cells", array=np.random.rand(10 * 10 * 10).ravel()),
            omf.NumericAttribute(name='attr4', location="cells", array=np.random.rand(10 * 10 * 10).ravel()),
        ]
    )

    project.elements = [block_model_1, block_model_2]

    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.omf')
    omf.save(project, temp_file.name, mode='w')
    return Path(temp_file.name)


# %%
# Read from block models
# ----------------------
# We'll demonstrate how to selectively merge attributes from each of the two block models to create a single dataframe.
# The `read_block_models` method accepts a dictionary where the keys are the block model names and the values are lists
# of attribute names to include in the dataframe.  If None is provided for the attribute list, all attributes
#

temp_omf_path = create_temp_omf_file()
reader = OMFPandasReader(temp_omf_path)

blockmodel_attributes = {
    'BlockModel1': None,
    'BlockModel2': ['attr4']
}

df = reader.read_block_models(blockmodel_attributes)
df.head(20)

# %%
# Check if the DataFrame contains the expected columns

expected_columns = ['attr1', 'attr2', 'attr4']
assert all(col in df.columns for col in expected_columns)

# %%
# Clean up temporary file

temp_omf_path.unlink()
