import pandas as pd
from pathlib import Path
from omfpandas.writer import OMFPandasWriter
from omfpandas.utils.blockmodel_utils import create_test_blockmodel


def test_write_regular_blockmodel(tmp_path):
    # Create a temporary OMF file path
    omf_file_path = tmp_path / "test.omf"

    # Create a test blockmodel DataFrame
    blocks = create_test_blockmodel(
        shape=(5, 4, 3), block_size=(1.0, 1.0, 0.5), corner=(100.0, 200.0, 300.0)
    )

    # Instantiate the OMFPandasWriter
    writer = OMFPandasWriter(filepath=omf_file_path)

    # Write the blockmodel to the OMF file
    writer.create_blockmodel(blocks, blockmodel_name="TestBlockModel")

    # Reload the project to confirm the blockmodel was written
    writer.persist_project()
    blockmodel = writer.get_element_by_name("TestBlockModel")

    # Assertions to confirm the blockmodel was written
    assert blockmodel is not None, "BlockModel was not created."
    assert blockmodel.name == "TestBlockModel", "BlockModel name does not match."
    assert (
        blockmodel.__class__.__name__ == "RegularBlockModel"
    ), "BlockModel type is not RegularBlockModel."


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
