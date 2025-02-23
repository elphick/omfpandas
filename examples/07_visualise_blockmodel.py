"""
Visualise a BlockModel
======================

It is always useful to visually validate our data. In this example, we will read a block model from an OMF file and
visualise it using PyVista.
"""
from pathlib import Path

import pandas as pd

from omfpandas import OMFPandasReader
from omfpandas.blockmodels.convert_blockmodel import convert_tensor_to_regular

# %%
# Instantiate
# -----------
# Create the object OMFPandas with the path to the OMF file.
test_omf_path: Path = Path('../assets/copper_deposit.omf')
omfp: OMFPandasReader = OMFPandasReader(filepath=test_omf_path)

# %%
# We'll inspect the elements in the omf file, and determine what volume element to convert.

omfp.element_types

# %%
# Read
# ----
# We can see by inspection that we have a TensorGridBlockModel in the omf file called *Block Model*, so we will
# convert that to a Pandas DataFrame, simply for the purposes of showing the first few records.

blocks: pd.DataFrame = omfp.read_blockmodel(blockmodel_name='Block Model', attributes=None)
print(f"DataFrame shape: {blocks.shape}")
blocks.head()

# %%
# We read the block model and convert it to an OMFBlockModel object

from omfpandas.blockmodel import OMFBlockModel

bm: OMFBlockModel = OMFBlockModel(omfp.get_element_by_name('Block Model'))

# %%
# Visualise
# ---------
# We can visualise the block model using PyVista.

p = bm.plot(scalar='CU_pct')
p.show()