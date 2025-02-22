"""
Validate BlockModel
===================

This example demonstrates how to validate a block model prior to writing it to an OMF file.

"""
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from omfpandas import OMFPandasReader, OMFPandasWriter

# %%
# Instantiate
# -----------
# Create the object OMFPandas with the path to the OMF file.

test_omf_path: Path = Path('../assets/test_file.omf')

# create a temporary copy to preserve the original file
temp_omf_path: Path = Path(tempfile.gettempdir()) / 'test_file_copy.omf'
shutil.copy(test_omf_path, temp_omf_path)

# Display the head of the original block model
blocks: pd.DataFrame = OMFPandasReader(filepath=temp_omf_path).read_blockmodel(blockmodel_name='regular')
blocks.head()

# %%
# Write with Schema
# -----------------
# The schema used to validate the dataframe is a Pandera schema.  The schema can be a YAML file (or Pandera compatible
# dictionary) that is loaded by the Pandera library.  The schema is used to validate the dataframe before writing it
# to the OMF file.

omfpw: OMFPandasWriter = OMFPandasWriter(filepath=temp_omf_path)

# %%
omfpw.create_blockmodel(blocks=blocks, blockmodel_name='regular',
                        pd_schema=test_omf_path.with_suffix('.schema.yaml'),
                        allow_overwrite=True)

# %%
# The schema is persisted inside the omf file.  This enables the schema to be used to validate modifications.

bm = omfpw.get_element_by_name('regular')
print(bm.metadata.get('pd_schema'))

# %%
# Demonstrate Failure
# -------------------
# We'll demonstrate a failure by modifying the block model in a way that violates the schema.

blocks['random attr'] = blocks['random attr'] * 2

try:
    omfpw.create_blockmodel(blocks=blocks, blockmodel_name='regular',
                            pd_schema=test_omf_path.with_suffix('.schema.yaml'),
                            allow_overwrite=True)
except Exception as e:
    print(e)

# %%
# Update with valid data

omfpw.write_blockmodel_attribute(blockmodel_name='regular', series=blocks['random attr'] / 4, allow_overwrite=True)

blocks: pd.DataFrame = OMFPandasReader(filepath=temp_omf_path).read_blockmodel(blockmodel_name='regular')
blocks.head()

# %%
# View the changelog
# ------------------

omfpw.changelog

# %%
# Clean up
temp_omf_path.unlink()
