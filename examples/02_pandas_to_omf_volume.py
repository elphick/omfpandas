"""
Pandas Block Model to OMF
=========================

This example demonstrates how to read a block model from an OMF file, modify it using pandas,
and write it back to a new OMF file.

"""

from pathlib import Path

import numpy as np
import omfvista
import pandas as pd
from omf import OMFReader, Project, VolumeElement

from omfpandas import OMFPandasReader, OMFPandasWriter
from omfpandas.volume import df_to_volume

# %%
# Read the block model
# --------------------
#
# Read the original block model from the OMF file.

test_omf_filepath: Path = Path('./../assets/test_file.omf')
blocks: pd.DataFrame = OMFPandasReader(filepath=test_omf_filepath).read_volume(volume_name='Block Model')
blocks.head()

# %%
# Write a new block model
# -----------------------
#
# We'll modify the block model and write it back to another OMF file.

blocks['density'] = 2.7
blocks['porosity'] = 0.1

new_omf_filepath: Path = Path('modified_test_file.omf')
# if not new_omf_file.exists():
#     new_omf_file.touch()

volume = df_to_volume(blocks, 'Modified Block Model')

OMFPandasWriter(filepath=new_omf_filepath).write_volume(blocks=blocks, volume_name='Modified Block Model',
                                                        allow_overwrite=True)

# %%
# Confirm that the new variables are in the model we saved.
saved_blocks: pd.DataFrame = OMFPandasReader(filepath=new_omf_filepath).read_volume(
    volume_name='Modified Block Model')
saved_blocks.head()

# %%
# Validation
# ----------
# Load the two omf projects and compare the CU_pct data for each

project_0: Project = OMFReader(str(test_omf_filepath)).get_project()
project_1: Project = OMFReader(str(new_omf_filepath)).get_project()

bm_1: VolumeElement = [element for element in project_0.elements if element.name == 'Block Model'][0]
bm_2: VolumeElement = [element for element in project_1.elements if element.name == 'Modified Block Model'][0]

bm_cu_1 = bm_1.data[0]
bm_cu_2 = bm_2.data[0]

assert np.all(bm_cu_1.name == bm_cu_2.name)

assert np.all(bm_cu_1.array.array == bm_cu_2.array.array)
