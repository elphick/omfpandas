"""
OMF Profile Block Model
=======================

Profiling a dataset is a common task in data analysis.  This example demonstrates how to profile an OMF block model.
The profile report is persisted inside the omf file.

"""
import logging
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from omfpandas import OMFPandasReader, OMFPandasWriter

# %%
# Instantiate
# -----------
# Create the object OMFPandas with the path to the OMF file.

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')
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

omfpw.write_block_model_schema(blockmodel_name='vol', pd_schema_filepath=test_omf_path.with_suffix('.schema.yaml'))
omfpw.profile_blockmodel(blockmodel_name='vol')
omfpw.view_block_model_profile(blockmodel_name='vol')

# %%
# Profile a subset with a query filter string

omfpw.profile_blockmodel(blockmodel_name='vol', query='`random attr`>0.5')
omfpw.view_block_model_profile(blockmodel_name='vol', query='`random attr`>0.5')
