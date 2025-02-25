import numpy as np
import pandas as pd

from omfpandas.utils import create_test_blockmodel


def test_create_regular_blockmodel():
    blocks: pd.DataFrame = create_test_blockmodel(shape=(5, 4, 3),
                                                  block_size=(1.0, 1.0, 0.5),
                                                  corner=(100.0, 200.0, 300.0))
    assert blocks.shape == (60, 3)
    assert blocks.index.names == ['x', 'y', 'z']

    assert np.all(blocks.index.get_level_values('x').unique() == [100.5, 101.5, 102.5, 103.5, 104.5])
    assert np.all(blocks.index.get_level_values('y').unique() == [200.5, 201.5, 202.5, 203.5])
    assert np.all(blocks.index.get_level_values('z').unique() == [300.25, 300.75, 301.25])

    assert blocks['c_style_xyz'].is_monotonic_increasing
    assert blocks.sort_index(level=['z', 'y', 'x'])['f_style_zyx'].is_monotonic_increasing

    # check the depth change
    assert np.all(np.diff(blocks.index.get_level_values('z').unique()) == 0.5)
    # check the absolute depth
    surface_rl = 300.0 + 3 * 0.5
    assert np.all(blocks.index.get_level_values('z').unique() == np.linspace(300.25, surface_rl - 0.25, 3))


def test_create_tensor_blockmodel():
    blocks: pd.DataFrame = create_test_blockmodel(shape=(5, 4, 3),
                                                  block_size=(1.0, 1.0, 0.5),
                                                  corner=(100.0, 200.0, 300.0),
                                                  is_tensor=True)
    assert blocks.shape == (60, 3)
    assert blocks.index.names == ['x', 'y', 'z', 'dx', 'dy', 'dz']

    assert np.all(blocks.index.get_level_values('x').unique() == [100.5, 101.5, 102.5, 103.5, 104.5])
    assert np.all(blocks.index.get_level_values('y').unique() == [200.5, 201.5, 202.5, 203.5])
    assert np.all(blocks.index.get_level_values('z').unique() == [300.25, 300.75, 301.25])
    assert np.all(blocks.index.get_level_values('dx').unique() == [1.0])
    assert np.all(blocks.index.get_level_values('dy').unique() == [1.0])
    assert np.all(blocks.index.get_level_values('dz').unique() == [0.5])

    assert blocks['c_style_xyz'].is_monotonic_increasing
    assert blocks.sort_index(level=['z', 'y', 'x'])['f_style_zyx'].is_monotonic_increasing

    # check the depth change
    assert np.all(np.diff(blocks.index.get_level_values('z').unique()) == 0.5)
    # check the absolute depth
    surface_rl = 300.0 + 3 * 0.5
    assert np.all(blocks.index.get_level_values('z').unique() == np.linspace(300.25, surface_rl - 0.25, 3))
