"""
OMF Block Model to Parquet
==========================

An omf VolumeElement represents a `Block Model`, and can be converted to a Parquet file.

.. note::
   Presently there is no low-memory option for this method, so it is not suitable for very large files.
   As such it offers no advantage over the standard Pandas method for saving to Parquet.
   However, this is the first step in a series of methods that will allow for more efficient handling of large files.

"""
from pathlib import Path

import pandas as pd

from omfpandas import OMFDataConverter, OMFPandasReader

# %%
# Instantiate
# -----------
# Create the object OMFPandas with the path to the OMF file.
test_omf_path: Path = Path('./../assets/test_file.omf')
omf_converter: OMFDataConverter = OMFDataConverter(filepath=test_omf_path)

# Display the head of the original block model
blocks: pd.DataFrame = OMFPandasReader(filepath=test_omf_path).read_volume(volume_name='Block Model')
print("Original DataFrame:")
print(blocks.head())

# %%
# Convert
# -------
# View the elements in the OMF file.
print(omf_converter.elements)

# Convert 'Block Model' to a Parquet file.
omf_converter.volume_to_parquet(volume_name='Block Model', parquet_filepath=Path('blocks.parquet'),
                                allow_overwrite=True)

# %%
# Load the Parquet
# ----------------
# Reload the Parquet file and display the head.

blocks_2: pd.DataFrame = pd.read_parquet('blocks.parquet')
print("Reloaded DataFrame:")
print(blocks_2.head())

# %%
# Validate
# --------
# Assert that the original DataFrame and the reloaded DataFrame are equivalent

assert blocks.equals(blocks_2), "The original DataFrame and the reloaded DataFrame are not equivalent."
