"""
Spatial Block ID
================

Each record in a tabular (pandas) block model represents a block in a 3D grid.  The columns represent the attributes
of the block, whicle the indes is the block identifier, typically x, y, z, being the block centroid in the
appropriate projection.

The pandas dataframe index being a multi-index with levels x, y, z, is a convenient way to represent the spatial
location of the block.  However this occupies 3 x 64-bit floats per block, which is wasteful.  The alternative is to
use a single integer to represent the block location.  This is the approach demonstrated here.

The integer is determined by bit-shifting the x, y, z values into a single 64-bit integer, reducing the memory
footprint by a factor of 3.  The integer can be used as a unique identifier for the block, and can be used to sort the
blocks in a way that is consistent with the spatial (c-major) ordering.

The integer can be converted back to x, y, z by bit-shifting in the opposite direction, with a convenience function.

"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

from omfpandas import OMFPandasWriter
from omfpandas.blockmodels.coordinate_encoding import encode_coordinates, multiindex_to_encoded_index, \
    encoded_index_to_multiindex
from omfpandas.utils import create_test_blockmodel

# %%
# Create Block Model Dataframe
# ----------------------------

shape = (5, 4, 3)
block_size = (1.0, 1.0, 0.5)
corner = (100.0, 200.0, 300.0)

blocks: pd.DataFrame = create_test_blockmodel(shape, block_size, corner)
multi_index = blocks.index

# %%
# The dataframe is C-style (x, y, z).  The last index (z) changes the fastest.

blocks

# %%
# Encode Spatial Block ID
# -----------------------
# Convert the index to a 64-bit integer.

blocks.index = multiindex_to_encoded_index(blocks.id)

blocks

# %%
# Decode Spatial Block ID
# -----------------------
# Decode the 64-bit integer back to x, y, z.

blocks.index = encoded_index_to_multiindex(blocks.index)
decoded_multi_index = blocks.index

blocks

# %%
# Verify the round trip
# ---------------------

# Verify the round-trip conversion
pd.testing.assert_index_equal(multi_index, decoded_multi_index, check_names=True)


# %%
# Sort Order Check
# ----------------
# Add a random attribute to support a sort order check. First check that the c-raveled attribute is monotonic.

assert np.array_equal(blocks.sort_index(level=['x', 'y', 'z'])['c_style_xyz'].values, np.arange(len(blocks)))

blocks['random_attr'] = np.random.rand(len(blocks))
