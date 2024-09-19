"""
Pandas Block Model to OMF
=========================

This example demonstrates how to read a block model from an OMF file, modify it using pandas,
and write it back to a new OMF file.

"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd
import omf
from omf import Project

from omfpandas import OMFPandasReader, OMFPandasWriter
from omfpandas.blockmodel import df_to_blockmodel

# %%
# Instantiate
# -----------

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')
test_omf_filepath: Path = Path('../assets/v2/test_file.omf')

# %%
# Read the block model
# --------------------
#
# Read the original block model from the OMF file.

blocks: pd.DataFrame = OMFPandasReader(filepath=test_omf_filepath).read_blockmodel(blockmodel_name='vol')
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

bm = df_to_blockmodel(blocks, 'Modified Block Model')

OMFPandasWriter(filepath=new_omf_filepath).write_blockmodel(blocks=blocks, blockmodel_name='Modified Block Model',
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
# Load the two omf projects and compare the CU_pct data for each

project_0: Project = omf.load(str(test_omf_filepath))
project_1: Project = omf.load(str(new_omf_filepath))

bm_1: omf.TensorGridBlockModel = [element for element in project_0.elements if element.name == 'vol'][0]
bm_2: omf.TensorGridBlockModel = [element for element in project_1.elements if element.name == 'Modified Block Model'][
    0]

bm_cu_1 = bm_1.attributes[0]
bm_cu_2 = bm_2.attributes[0]

assert np.all(bm_cu_1.name == bm_cu_2.name)

assert np.all(bm_cu_1.array.array == bm_cu_2.array.array)
