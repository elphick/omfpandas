"""
Create OMF2 File
================

This script creates a test omf2 file containing two versions of the same model:
vol: Regular
tensor: TensorGrid

We start with an existing file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import omf
from omf import Project, TensorGridBlockModel

from omfpandas import OMFPandasReader, OMFPandasWriter

# %%
# Read
# ----
# Read the original block model from the OMF file which is a tensor model


test_omf_filepath: Path = Path('../assets/v2/test_file.original.omf')
omfpr: OMFPandasReader = OMFPandasReader(filepath=test_omf_filepath)
blocks: pd.DataFrame = omfpr.read_blockmodel(blockmodel_name='vol')
blocks.head()

# %%
# Write
# -----
# Write the same model as a regular model, achieved by dropping the dx, dy,, dz index levels.
# This is only possible because we know the tensor model is regular also (uniform block size throughout).

omfpw: OMFPandasWriter = OMFPandasWriter(test_omf_filepath.with_suffix('.modified.omf'))
omfpw.create_blockmodel(blocks, blockmodel_name='tensor', allow_overwrite=True)
blocks.index = blocks.index.droplevel(['dx', 'dy', 'dz'])
omfpw.create_blockmodel(blocks, blockmodel_name='regular', allow_overwrite=True)

# %%
# append the original objects
for el in omfpr.project.elements:
    if not isinstance(el, TensorGridBlockModel):
        omfpw.project.elements.append(el)

print([el.name for el in omfpw.project.elements])