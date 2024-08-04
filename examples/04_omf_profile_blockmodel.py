"""
OMF Profile Block Model
=======================

Profiling a dataset is a common task in data analysis.  This example demonstrates how to profile an OMF block model.
The profile report is persisted inside the omf file.

"""
import shutil
import tempfile
import webbrowser
from pathlib import Path

import pandas as pd

from omfpandas import OMFDataConverter, OMFPandasReader, OMFPandasWriter

# %%
# Instantiate
# -----------
# Create the object OMFPandas with the path to the OMF file.
test_omf_path: Path = Path('./../assets/v2/test_file.omf')

# create a temporary copy to preserve the original file
temp_omf_path: Path = Path(tempfile.gettempdir()) / 'test_file_copy.omf'
shutil.copy(test_omf_path, temp_omf_path)

# Display the head of the original block model
blocks: pd.DataFrame = OMFPandasReader(filepath=temp_omf_path).read_blockmodel(blockmodel_name='vol')
blocks.head()

# %%
# Profile
# -------
# View the elements in the OMF file first.

omfpw: OMFPandasWriter = OMFPandasWriter(filepath=temp_omf_path)

omfpw.profile_blockmodel(blockmodel_name='vol')
omfpw.view_block_model_profile(blockmodel_name='vol')

# %%
# Profile a subset with a query filter string

omfpw.profile_blockmodel(blockmodel_name='vol', query='`random attr`>0.5')
omfpw.view_block_model_profile(blockmodel_name='vol', query='`random attr`>0.5')

