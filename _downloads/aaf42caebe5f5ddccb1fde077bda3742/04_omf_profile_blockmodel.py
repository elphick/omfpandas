"""
OMF Profile Block Model
=======================

Profiling a dataset is a common task in data analysis.  This example demonstrates how to profile an OMF block model.
The profile report is persisted inside the omf file.

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
# Profile
# -------
# Create the writer, write the pandera schema and the profile report into the file.
# The use of a pandera schema is optional, but it provides a way to describe the attributes in the dataset.

omfpw: OMFPandasWriter = OMFPandasWriter(filepath=temp_omf_path)
omfpw.write_block_model_schema(blockmodel_name='regular', pd_schema_filepath=test_omf_path.with_suffix('.schema.yaml'))
omfpw.profile_blockmodel(blockmodel_name='regular')

# %%
# View the profile report, which benefits from the attribute descriptions from the schema.
omfpw.view_block_model_profile(blockmodel_name='regular')

# %%
# Profile a subset with a query filter string

omfpw.profile_blockmodel(blockmodel_name='regular', query='`random attr`>0.5')

# %%
# View the profile report of the subset.  The dataset tab in the profile report describes the filter applied to the
# dataset.

omfpw.view_block_model_profile(blockmodel_name='regular', query='`random attr`>0.5')
