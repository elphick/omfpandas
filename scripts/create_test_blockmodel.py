"""
Create a test blockmodel that can be used to test the omfpandas package, particularly in relation to block ordering.

"""

import pandas as pd
import numpy as np

from omfpandas import OMFPandasWriter
from omfpandas.blockmodel import OMFBlockModel

# Define the block model parameters
shape = (5, 4, 3)
block_size = (1.0, 1.0, 0.5)
num_blocks = np.prod(shape)
origin = (100.0, 200.0, 300.0)

# Generate the coordinates for the block model
x_coords = np.arange(origin[0] + block_size[0] / 2, origin[0] + shape[0] * block_size[0], block_size[0])
y_coords = np.arange(origin[1] + block_size[1] / 2, origin[1] + shape[1] * block_size[1], block_size[1])
z_coords = np.arange(origin[2] + block_size[2] / 2, origin[2] + shape[2] * block_size[2], block_size[2])

# Create a meshgrid of coordinates
xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

# Flatten the coordinates
xx_flat_c = xx.ravel(order='C')
yy_flat_c = yy.ravel(order='C')
zz_flat_c = zz.ravel(order='C')

xx_flat_f = xx.ravel(order='F')
yy_flat_f = yy.ravel(order='F')
zz_flat_f = zz.ravel(order='F')

# Create the attributes
c_order_xyz = np.arange(num_blocks)
f_order_zyx = np.arange(num_blocks)

depth = (origin[2] + shape[2] * block_size[2] - zz_flat_c) / block_size[2]

# Create the DataFrame
df_c = pd.DataFrame({
    'x': xx_flat_c,
    'y': yy_flat_c,
    'z': zz_flat_c,
    'c_order_xyz': c_order_xyz})

df_f = pd.DataFrame({
    'x': xx_flat_f,
    'y': yy_flat_f,
    'z': zz_flat_f,
    'f_order_zyx': f_order_zyx})

# Set the index to x, y, z
df_c.set_index(keys=['x', 'y', 'z'], inplace=True)
df_f.set_index(keys=['x', 'y', 'z'], inplace=True)
df = pd.concat([df_c, df_f], axis=1)
df.sort_index(level=['x', 'y', 'z'], inplace=True)
df['depth'] = depth


print(df)
print(df.sort_index(level=['z', 'y', 'x']))

# Check the ordering - confirm that the c_order_xyz and f_order_zyx columns are in the correct order
assert np.array_equal(df.sort_index(level=['x', 'y', 'z'])['c_order_xyz'].values, np.arange(num_blocks))
assert np.array_equal(df.sort_index(level=['z', 'y', 'x'])['f_order_zyx'].values, np.arange(num_blocks))

# Write the DataFrame to a parquet file
df.to_parquet('test_blockmodel.parquet')

# %%
# Create OMF
# ----------

omfpw: OMFPandasWriter = OMFPandasWriter(filepath='test_blockmodel.omf')
omfpw.create_blockmodel(blocks=df, blockmodel_name='sort_check', allow_overwrite=True)

# %%
# Visualise
# ---------

p = OMFBlockModel(omfpw.get_element_by_name('sort_check')).plot(scalar='depth')
p.show()

p = OMFBlockModel(omfpw.get_element_by_name('sort_check')).plot(scalar='c_order_xyz')
p.show()

p = OMFBlockModel(omfpw.get_element_by_name('sort_check')).plot(scalar='f_order_zyx')
p.show()
