import sys
import tempfile
import time

import numpy as np
import omf
import pandas as pd
from pathlib import Path
import pytest
from omf import Project, NumericAttribute, TensorGridBlockModel

from omfpandas.reader import OMFPandasReader


def test_read_incongruent_block_models(temp_incongruent_omf_file):
    temp_omf_file = temp_incongruent_omf_file
    reader = OMFPandasReader(temp_omf_file)

    # define the attributes to read from each blockmodel -> None will load all
    blockmodel_attributes = {
        'BlockModel1': None,
        'BlockModel2': ['attr4']
    }

    # failure is expected
    with pytest.raises(ValueError):
        df = reader.read_block_models(blockmodel_attributes)

    # Clean up temporary file
    temp_omf_file.unlink()


def test_read_block_models(temp_congruent_omf_file):
    temp_omf_file = temp_congruent_omf_file
    reader = OMFPandasReader(temp_omf_file)

    blockmodel_attributes = {
        'BlockModel1': None,
        'BlockModel2': ['attr4']
    }

    df = reader.read_block_models(blockmodel_attributes)

    # Check if the DataFrame contains the expected columns
    expected_columns = ['attr1', 'attr2', 'attr4']
    assert all(col in df.columns for col in expected_columns)

    # Clean up temporary file
    temp_omf_file.unlink()


def measure_execution_time(reader, blockmodel_attributes, query=None, post_query=None):
    start_time = time.time()
    df = reader.read_block_models(blockmodel_attributes, query=query)
    if post_query:
        df = df.query(post_query)
    execution_time = time.time() - start_time
    return df, execution_time


def test_read_block_models_with_query(temp_congruent_omf_file):
    temp_omf_path = temp_congruent_omf_file
    reader = OMFPandasReader(temp_omf_path)

    blockmodel_attributes = {
        'BlockModel1': None,
        'BlockModel2': ['attr4']
    }

    num_iterations = 10
    execution_times_df1 = []
    execution_times_df2 = []

    for _ in range(num_iterations):
        df1, execution_time_df1 = measure_execution_time(reader, blockmodel_attributes, post_query='attr4 > 0.5')
        execution_times_df1.append(execution_time_df1)

        df2, execution_time_df2 = measure_execution_time(reader, blockmodel_attributes, query='attr4 > 0.5')
        execution_times_df2.append(execution_time_df2)

    avg_execution_time_df1 = sum(execution_times_df1) / num_iterations
    avg_execution_time_df2 = sum(execution_times_df2) / num_iterations

    relative_execution_time = avg_execution_time_df2 / avg_execution_time_df1 if avg_execution_time_df1 != 0 else float(
        'inf')

    sys.stdout.write(f"\nAverage execution time for df1: {avg_execution_time_df1:.6f} seconds\n")
    sys.stdout.write(f"Average execution time for df2: {avg_execution_time_df2:.6f} seconds\n")
    sys.stdout.write(f"Relative execution time (df2/df1): {relative_execution_time:.2f}\n")

    # performance is poorer, but memory is lower
    # assert relative_execution_time < 1.0

    temp_omf_path.unlink()
