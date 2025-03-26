"""
OMF Block Model to Parquet
==========================

A parquet file is a columnar storage format, enabling column by column reading and writing.  This can be used
to reduce memory consumption.  This example demonstrates how to convert an OMF block model to a Parquet file.

.. note::
   Presently there is no low-memory option for this method, so it is not suitable for very large files.
   As such it offers no advantage over the standard Pandas method for saving to Parquet.
   However, this is the first step in a series of methods that will allow for more efficient handling of large files.

"""
from pathlib import Path

import pandas as pd

from omfpandas import OMFPandasReader, OMFPandasWriter

# %%
# Instantiate
# -----------
# Create the object OMFPandas with the path to the OMF file.

test_omf_path: Path = Path('../assets/test_file.omf')
omf_writer: OMFPandasWriter = OMFPandasWriter(filepath=test_omf_path)

# Display the head of the original block model
blocks: pd.DataFrame = OMFPandasReader(filepath=test_omf_path).read_blockmodel(blockmodel_name='regular')
print("Original DataFrame:")
blocks.head()

# %%
# Convert
# -------
# View the elements in the OMF file first.
print(omf_writer.element_types)

# %%
# Convert 'Block Model' to a Parquet file.
omf_writer.blockmodel_to_parquet(blockmodel_name='regular', out_path=Path('blocks.parquet'),
                                 allow_overwrite=True)

# %%
# Load the Parquet
# ----------------
# Reload the Parquet file and display the head.

blocks_2: pd.DataFrame = pd.read_parquet('blocks.parquet')
print("Reloaded DataFrame:")
blocks_2.head()

# %%
# Validate
# --------
# Assert that the original DataFrame and the reloaded DataFrame are equivalent

assert blocks.equals(blocks_2)
