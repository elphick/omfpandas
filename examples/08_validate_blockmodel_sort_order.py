"""
Validate BlockModel Sort Order
==============================

A block model is a 3D array of cells, each with a set of attributes.  However, it is convenient to represent the
attributes as a vector.  The order of the data in the vector is determined by the order used in this (ravel) conversion.

The ravel order for OMF is C-style (row-major order).
The ravel order for Pandas is C-style (row-major order).
The ravel order for PyVista is F-style (column-major order).

- C-style - row-major order:  The last index (z) changes the fastest.  The first index (x) changes the slowest. :code:`df.sort_index(['x', 'y', 'z'])`
- F-style - column-major order:  The first index (x) changes the fastest.  The last index (z) changes the slowest. :code:`df.sort_index(['z', 'y', 'x'])`

This script completes some validations on the sort order of the block model data:

- parquet file -> dataframe
- dataframe -> omf file / regular block model
- omf file / regular block model -> dataframe
- pyvista visualisation

To conduct this test we use a small block model of shape (5, 4, 3) with a depth attribute.

So we will round-trip the data from the dataframe to the omf file, and back to a dataframe.
We will also check the visualisation performs as expected.

"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

from omfpandas import OMFPandasWriter

# %%
# Load
# ----
# Create the object OMFPandas with the path to the OMF file.
test_bm_path: Path = Path('../assets/test_blockmodel.parquet')
blocks: pd.DataFrame = pd.read_parquet(test_bm_path)

# %%
# The dataframe is C-style (x, y, z).  The last index (z) changes the fastest.

blocks

# %%
# Sort to F-style (z, y, x).  The first index (x) changes the fastest.
blocks.sort_index(level=['z', 'y', 'x'])

# %%
# Sort Order Check
# ----------------
# Check the ordering by first sorting by x,y,z and confirming the c-raveled attribute is monotonic.
assert np.array_equal(blocks.sort_index(level=['x', 'y', 'z'])['c_order_xyz'].values, np.arange(len(blocks)))

# %%
# Similarly, sort by z,y,x and confirm the f-raveled attribute is monotonic.
assert np.array_equal(blocks.sort_index(level=['z', 'y', 'x'])['f_order_zyx'].values, np.arange(len(blocks)))


# %%
# Create OMF
# ----------

omfp: OMFPandasWriter = OMFPandasWriter(filepath='test_blockmodel.omf')
omfp.create_blockmodel(blocks=blocks, blockmodel_name='sort_check', allow_overwrite=True)

# %%
# Export to Pandas
# ----------------
# Check the round trip: pandas > omf > pandas

blocks_omf = omfp.read_blockmodel(blockmodel_name='sort_check')

pd.testing.assert_frame_equal(blocks, blocks_omf)

# %%
# Visualise
# ---------
# Plot the depth attribute to confirm the sort order is as expected.
p = omfp.plot_blockmodel(blockmodel_name='sort_check', scalar='depth', threshold=False)
p.show()

# %%
# Plot the c-raveled attribute.

p: pv.Plotter = omfp.plot_blockmodel(blockmodel_name='sort_check', scalar='c_order_xyz', threshold=False)
p.show()

# %%
# Plot the f-raveled attribute.

p = omfp.plot_blockmodel(blockmodel_name='sort_check', scalar='f_order_zyx', threshold=False)
p.show()
