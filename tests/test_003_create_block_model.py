import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import yaml

from omfpandas import OMFPandasWriter
from conftest import get_test_schema


def create_dataframe_blockmodel(origin: tuple[float, float, float] = (0., 0., 0.),
                                shape: tuple[int, int, int] = (10, 10, 5),
                                cell_size: tuple[float, float, float] = (1., 1., 1.)) -> pd.DataFrame:
    num_cells: int = math.prod(shape)

    # create a dataframe with synthetic data for a blockmodel with dims (10, 10, 5) having cell sizes of 1x1x1
    index = pd.MultiIndex.from_product([np.arange(origin[0], shape[0] + origin[0], cell_size[0]),
                                        np.arange(origin[1], shape[1] + origin[1], cell_size[1]),
                                        np.arange(origin[2], shape[2] + origin[2], cell_size[2])],
                                       names=['x', 'y', 'z'])
    # add the dx, dy, dz to the index for all blocks
    index = pd.concat(
        [index.to_frame(index=False),
         pd.DataFrame(data={'dx': cell_size[0], 'dy': cell_size[1], 'dz': cell_size[1]},
                      index=np.arange(num_cells))],
        axis=1).set_index(['x', 'y', 'z', 'dx', 'dy', 'dz']).index

    blocks = pd.DataFrame(index=index,
                          data={'attr1': np.random.rand(num_cells), 'attr2': np.random.rand(num_cells)})
    return blocks


def test_create_blockmodel_from_df_no_schema():
    blocks: pd.DataFrame = create_dataframe_blockmodel()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / 'blockmodel.omf'
        writer: OMFPandasWriter = OMFPandasWriter(filepath=temp_file_path)
        writer.create_blockmodel(blocks=blocks, blockmodel_name='Block Model', allow_overwrite=True)

        assert os.path.exists(temp_file_path)
        assert temp_file_path.stat().st_size > 0
        assert writer.project.elements[0].name == 'Block Model'
        assert writer.project.elements[0].attributes[0].name == 'attr1'
        assert writer.project.elements[0].attributes[1].name == 'attr2'
        assert writer.project.elements[0].description == ''

def test_create_blockmodel_from_df_with_schema_file():
    blocks: pd.DataFrame = create_dataframe_blockmodel()
    schema_file: Path = get_test_schema()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / 'blockmodel.omf'
        writer: OMFPandasWriter = OMFPandasWriter(filepath=temp_file_path)
        writer.create_blockmodel(blocks=blocks, blockmodel_name='Block Model', pd_schema=schema_file,
                                 allow_overwrite=True)

        assert os.path.exists(temp_file_path)
        assert temp_file_path.stat().st_size > 0
        assert writer.project.elements[0].name == 'Block Model'
        assert writer.project.elements[0].attributes[0].name == 'attr1'
        assert writer.project.elements[0].attributes[1].name == 'attr2'
        assert writer.project.elements[0].description == 'A test dataset schema.'

def test_create_blockmodel_from_df_with_schema_dict():
    blocks: pd.DataFrame = create_dataframe_blockmodel()
    schema_file: Path = get_test_schema()

    schema_dict: dict = yaml.safe_load(schema_file.read_text())
    schema_dict['description'] = 'A modified test dataset schema.'

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / 'blockmodel.omf'
        writer: OMFPandasWriter = OMFPandasWriter(filepath=temp_file_path)
        writer.create_blockmodel(blocks=blocks, blockmodel_name='Block Model', pd_schema=schema_dict,
                                 allow_overwrite=True)

        assert os.path.exists(temp_file_path)
        assert temp_file_path.stat().st_size > 0
        assert writer.project.elements[0].name == 'Block Model'
        assert writer.project.elements[0].attributes[0].name == 'attr1'
        assert writer.project.elements[0].attributes[1].name == 'attr2'
        assert writer.project.elements[0].description == 'A modified test dataset schema.'
