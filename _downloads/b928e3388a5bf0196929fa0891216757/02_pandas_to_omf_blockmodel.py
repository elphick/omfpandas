"""
Pandas Block Model to OMF
=========================

This example demonstrates how to read a block model from an OMF file, modify it using pandas,
and write it back to a new OMF file.  A round trip is completed to validate that the omf file stores various
data types as expected.

"""

from pathlib import Path

import numpy as np
import pandas as pd
import omf
from omf import Project

from omfpandas import OMFPandasReader, OMFPandasWriter

# %%
# Instantiate
# -----------
# Read the original block model from the OMF file.


test_omf_filepath: Path = Path('../assets/test_file.omf')
blocks: pd.DataFrame = OMFPandasReader(filepath=test_omf_filepath).read_blockmodel(blockmodel_name='regular')
blocks.head()

# %%
# Modify the BlockModel
# ---------------------
#
# We'll modify the block model using Pandas and write it back to another OMF file.
# Add a new float column to the block model.

blocks['density'] = 2.7
# set the second density record to nan
blocks.iloc[1, blocks.columns.get_loc('density')] = np.nan

# %%
# Create an integer variable

np.random.seed(42)
blocks['some_int'] = pd.Series(np.random.choice([0, 1, 2, np.nan], size=blocks.shape[0], ), dtype='Int32',
                               index=blocks.index)

# %%
# Create a categorical variable

blocks['rock_class'] = np.random.choice(['A', 'B', 'C', np.nan], size=blocks.shape[0])
blocks['rock_class'] = blocks['rock_class'].astype('category')
blocks.head()

# %%
blocks.dtypes

# %%
# Write to OMF
# ------------

new_omf_filepath: Path = Path('modified_test_file.omf')

OMFPandasWriter(filepath=new_omf_filepath).create_blockmodel(blocks=blocks, blockmodel_name='Modified Block Model',
                                                             allow_overwrite=True)
omfpr: OMFPandasReader = OMFPandasReader(filepath=new_omf_filepath)
# %%
# Confirm that the new variables are in the model we saved.
saved_blocks: pd.DataFrame = OMFPandasReader(filepath=new_omf_filepath).read_blockmodel(
    blockmodel_name='Modified Block Model')
saved_blocks.head()

# %%
saved_blocks.dtypes

# %%
# Validation
# ----------
# Load the two omf projects and compare the attribute data for each

project_1: Project = omf.load(str(test_omf_filepath))
project_2: Project = omf.load(str(new_omf_filepath))

bm_1: omf.RegularBlockModel = [element for element in project_1.elements if element.name == 'regular'][0]
bm_2: omf.RegularBlockModel = [element for element in project_2.elements if element.name == 'Modified Block Model'][0]

for i, attr_1 in enumerate(bm_1.attributes):
    attr_2 = bm_2.attributes[i]
    assert np.all(attr_1.name == attr_2.name)
    assert np.all(attr_1.array.array == attr_2.array.array)
