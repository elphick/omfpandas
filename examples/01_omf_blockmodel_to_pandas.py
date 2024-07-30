"""
OMF Block Model to DataFrame
============================

An omf VolumeElement represents a `Block Model`, and can be converted to a Pandas DataFrame.

"""
from pathlib import Path

import pandas as pd

from omfpandas import OMFPandasReader

# %%
# Instantiate
# -----------
# Create the object OMFPandas with the path to the OMF file.
test_omf_path: Path = Path('../assets/v2/test_file.omf')
omfp: OMFPandasReader = OMFPandasReader(filepath=test_omf_path)

# %%
# Convert
# -------
# We'll inspect the elements in the omf file, and determine what volume element to convert.

omfp.elements

# %%
# We can see by inspection that we have one volume element in the omf file called 'Block Model, so we will
# convert that to a Pandas DataFrame.

blocks: pd.DataFrame = omfp.read_blockmodel(blockmodel_name='vol', variables=None, with_geometry_index=True)
print(f"DataFrame shape: {blocks.shape}")
blocks.head()

# %%
# The index contains the centroid coordinates and the dimensions of the block.
# The columns contain the variables in the block model, though only variables (attributes) assigned to the `cell`
# (as distinct from the grid `points`) are loaded.

