"""
Block Model to DataFrame
========================

An omf VolumeElement represents a `Block Model`, and can be converted to a Pandas DataFrame.

"""
from pathlib import Path
import pandas as pd

from omfpandas import OMFPandas

# %%
# Instantiate
# -----------
# Create the object OMFPandas with the path to the OMF file.

omfp: OMFPandas = OMFPandas(Path('./../assets/test_file.omf'))

# %%
# Convert
# -------
# We'll inspect the elements in the omf file, and determine what volume element to convert.

omfp.elements

# %%
# We can see by inspection that we have one volume element in the omf file called 'Block Model, so we will
# convert that to a Pandas DataFrame.

blocks: pd.DataFrame = omfp.volume_to_df(volume_name='Block Model', variables=None, with_geometry_index=True)
print(f"DataFrame shape: {blocks.shape}")
blocks.head()

# %%
# The index contains the centroid coordinates and the dimensions of the block.
# The columns contain the variables in the block model, though only variables assigned to the `cell`
# (as distinct from the grid `points`) are loaded.

# %%
# Save to Parquet
# ---------------
# Of course we can save the DataFrame to a Parquet file in the usual way.
#
# .. code::
#
#    blocks.to_parquet('blocks.parquet')
#
# An alternative method is to use the `volume_to_parquet` method of the OMFPandas object.
# This method will save the DataFrame to a Parquet file.  In later versions, this method will have a low-memory
# option to handle large files.

omfp.volume_to_parquet(volume_name='Block Model', parquet_filepath=Path('blocks.parquet'))

# %%
# Load the Parquet
# ----------------

blocks_2: pd.DataFrame = pd.read_parquet('blocks.parquet')
print(f"DataFrame shape: {blocks_2.shape}")
blocks_2.head()
