from pathlib import Path

import numpy as np
import pandas as pd

from omfpandas import OMFPandasWriter
from omfpandas.utils import create_test_blockmodel
from conftest import get_test_schema


def test_write_regular_blockmodel_to_composite(tmp_path):
    # Create a temporary OMF file path
    omf_file_path = tmp_path / "test.omf"

    # Create a test blockmodel DataFrame
    blocks = create_test_blockmodel(
        shape=(5, 4, 3), block_size=(1.0, 1.0, 0.5), corner=(100.0, 200.0, 300.0)
    )

    # Instantiate the OMFPandasWriter
    writer = OMFPandasWriter(filepath=omf_file_path)

    # Write the blockmodel to a composite using dot notation
    blockmodel_name = "TestComposite.TestBlockModel"
    writer.create_blockmodel(blocks, blockmodel_name=blockmodel_name)

    # Reload the project to confirm the blockmodel was written
    writer.persist_project()
    composite = writer.get_element_by_name("TestComposite")
    blockmodel = writer.get_element_by_name("TestComposite.TestBlockModel")

    # Assertions to confirm the blockmodel was written to the composite
    assert composite is not None, "Composite was not created."
    assert blockmodel is not None, "BlockModel was not created."
    assert (
        blockmodel in composite.elements
    ), "BlockModel was not added to the composite."
    assert blockmodel.name == "TestBlockModel", "BlockModel name does not match."
    assert (
        blockmodel.__class__.__name__ == "RegularBlockModel"
    ), "BlockModel type is not RegularBlockModel."


def test_composite_with_calculated_attribute(tmp_path):
    blocks: pd.DataFrame = create_test_blockmodel(shape=(5, 4, 3), block_size=(1.0, 1.0, 0.5),
                                                  corner=(0.0, 0.0, 0.0), is_tensor=False)
    blocks['attr1'] = np.random.rand(5 * 4 * 3)
    blocks['attr2'] = np.random.rand(5 * 4 * 3)

    # Instantiate the OMFPandasWriter
    writer = OMFPandasWriter(filepath= Path(tmp_path) / 'test.omf')
    writer.create_blockmodel(blocks=blocks, blockmodel_name='composite.regular_blockmodel',
                             pd_schema=get_test_schema())

    # Read the blockmodel back in
    blocks_out: pd.DataFrame = writer.read_blockmodel('composite.regular_blockmodel')

    # assert the calculated attributes are valid
    pd.testing.assert_series_equal(blocks_out['calc_attr1'], blocks_out['attr1'] * 2, check_names=False)
    pd.testing.assert_series_equal(blocks_out['calc_attr2'], blocks_out['attr1'], check_names=False)


