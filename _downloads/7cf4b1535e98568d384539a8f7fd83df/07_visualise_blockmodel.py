"""
Visualise a BlockModel
======================

It is always useful to visually validate our data. In this example, we will read a block model from an OMF file and
visualise it using PyVista.
"""
import shutil
from pathlib import Path

import pandas as pd

from omfpandas import OMFPandasReader, OMFPandasWriter
from omfpandas.blockmodels.convert_blockmodel import convert_tensor_to_regular, df_to_tensor_bm, df_to_regular_bm, \
    blockmodel_to_df

# %%
# Instantiate
# -----------
# Create the object OMFPandas with the path to the OMF file.
test_omf_path: Path = Path('../assets/copper_deposit.omf')
omfpr: OMFPandasReader = OMFPandasReader(filepath=test_omf_path)

# %%
# We'll inspect the elements in the omf file, and determine what volume element to convert.

omfpr.element_types

# %%
# Read
# ----
# We can see by inspection that we have a TensorGridBlockModel in the omf file called *Block Model*, so we will
# convert that to a Pandas DataFrame, simply for the purposes of showing the first few records.

blocks: pd.DataFrame = omfpr.read_blockmodel(blockmodel_name='Block Model', attributes=None)
print(f"DataFrame shape: {blocks.shape}")
blocks.head()

# %%
# We read the block model and convert it to an OMFBlockModel object

from omfpandas.blockmodel import OMFBlockModel

bm: OMFBlockModel = OMFBlockModel(omfpr.get_element_by_name('Block Model'))

# %%
# Visualise
# ---------
# We can visualise the block model using PyVista.

p = bm.plot(scalar='CU_pct')
p.show()

# %%
# Create a regular model

# ----------------------
# We can only do this since this particular tensor model has consistent block sizes.
#
# Make a copy of the file first.

demo_omf_filepath: Path = shutil.copy2(test_omf_path, test_omf_path.with_suffix('.modified.omf'))

regular_bm = df_to_regular_bm(blockmodel_name='Regular Block Model', df=blocks.droplevel(level=['dx', 'dy', 'dz']))
regular_blocks: pd.DataFrame = blockmodel_to_df(regular_bm)

omfpw: OMFPandasWriter = OMFPandasWriter(filepath=demo_omf_filepath)
omfpw.create_blockmodel(blocks=regular_blocks, blockmodel_name='Regular Block Model')

# %%
# Visualise Regular Model
# -----------------------

bm_regular: OMFBlockModel = OMFBlockModel(omfpw.get_element_by_name('Regular Block Model'))
p = bm_regular.plot(scalar='CU_pct')
p.show()
