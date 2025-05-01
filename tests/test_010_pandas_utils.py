import pandas as pd
import numpy as np
import pytest
from omfpandas.utils.pandas_utils import aggregate


def test_aggregate_majority():
    # Create a sample DataFrame
    data = {
        'dry_mass': [1, 2, 3, 4],
        'assay1': [10, 20, 30, 40],
        'assay2': [100, 200, 300, 400],
        'other': [5, 6, 7, 8],
        'category': pd.Categorical(['a', 'a', 'b', 'a'])
    }
    df = pd.DataFrame(data)

    # Define the aggregation dictionary
    agg_dict = {
        'assay1': 'dry_mass',
        'assay2': 'dry_mass'
    }

    # Expected result for majority category
    expected_data_majority = {
        'dry_mass': 10,
        'assay1': np.average([10, 20, 30, 40], weights=[1, 2, 3, 4]),
        'assay2': np.average([100, 200, 300, 400], weights=[1, 2, 3, 4]),
        'other': 26,
        'category': 'a'  # Majority category
    }
    expected_df_majority = pd.DataFrame([expected_data_majority])

    # Run the aggregate function with majority category treatment
    result_df_majority = aggregate(df, agg_dict, cat_treatment='majority')

    # Ensure the columns are in the same order
    pd.testing.assert_frame_equal(result_df_majority, expected_df_majority)


def test_aggregate_proportions():
    # Create a sample DataFrame
    data = {
        'dry_mass': [1, 2, 3, 4],
        'assay1': [10, 20, 30, 40],
        'assay2': [100, 200, 300, 400],
        'other': [5, 6, 7, 8],
        'category': pd.Categorical(['a', 'a', 'b', 'a'])
    }
    df = pd.DataFrame(data)

    # Define the aggregation dictionary
    agg_dict = {
        'assay1': 'dry_mass',
        'assay2': 'dry_mass'
    }

    # Expected result for category proportions
    expected_data_proportions = {
        'dry_mass': 10,
        'assay1': np.average([10, 20, 30, 40], weights=[1, 2, 3, 4]),
        'assay2': np.average([100, 200, 300, 400], weights=[1, 2, 3, 4]),
        'other': 26,
        'category': {'a': 0.75, 'b': 0.25}  # Proportions of each category
    }
    expected_df_proportions = pd.DataFrame([expected_data_proportions])

    # Run the aggregate function with category proportions treatment
    result_df_proportions = aggregate(df, agg_dict, cat_treatment='proportions')

    # Ensure the columns are in the same order
    pd.testing.assert_frame_equal(result_df_proportions, expected_df_proportions)


def test_aggregate_proportions_as_columns():
    # Create a sample DataFrame
    data = {
        'dry_mass': [1, 2, 3, 4],
        'assay1': [10, 20, 30, 40],
        'assay2': [100, 200, 300, 400],
        'other': [5, 6, 7, 8],
        'category': pd.Categorical(['a', 'a', 'b', 'a'])
    }
    df = pd.DataFrame(data)

    # Define the aggregation dictionary
    agg_dict = {
        'assay1': 'dry_mass',
        'assay2': 'dry_mass'
    }

    # Expected result for category proportions as columns
    expected_data_proportions_as_columns = {
        'dry_mass': 10,
        'assay1': np.average([10, 20, 30, 40], weights=[1, 2, 3, 4]),
        'assay2': np.average([100, 200, 300, 400], weights=[1, 2, 3, 4]),
        'other': 26,
        'category_a': 0.75,  # Proportion of category 'a'
        'category_b': 0.25  # Proportion of category 'b'
    }
    expected_df_proportions_as_columns = pd.DataFrame([expected_data_proportions_as_columns])

    # Run the aggregate function with category proportions as columns treatment
    result_df_proportions_as_columns = aggregate(df, agg_dict, cat_treatment='proportions', proportions_as_columns=True)

    # Ensure the columns are in the same order
    pd.testing.assert_frame_equal(result_df_proportions_as_columns, expected_df_proportions_as_columns)


def test_aggregate_with_groupby():
    # Create a sample DataFrame
    data = {
        'group': pd.Categorical(['A', 'A', 'B', 'B']),
        'dry_mass': [1, 2, 3, 4],
        'assay1': [10, 20, 30, 40],
        'assay2': [100, 200, 300, 400],
        'other': [5, 6, 7, 8],
        'category': pd.Categorical(['a', 'a', 'b', 'a'])
    }
    df = pd.DataFrame(data)

    # Define the aggregation dictionary
    agg_dict = {
        'assay1': 'dry_mass',
        'assay2': 'dry_mass'
    }

    # Expected result for group 'A'
    expected_data_group_A = {
        'group': 'A',
        'dry_mass': 3,
        'assay1': np.average([10, 20], weights=[1, 2]),
        'assay2': np.average([100, 200], weights=[1, 2]),
        'other': 11,
        'category': 'a'  # Majority category
    }
    expected_df_group_A = pd.DataFrame([expected_data_group_A])

    # Expected result for group 'B'
    expected_data_group_B = {
        'group': 'B',
        'dry_mass': 7,
        'assay1': np.average([30, 40], weights=[3, 4]),
        'assay2': np.average([300, 400], weights=[3, 4]),
        'other': 15,
        'category': 'a'  # Majority category
    }
    expected_df_group_B = pd.DataFrame([expected_data_group_B])

    # Group by 'group' and apply the aggregate function
    result_df = df.groupby('group').apply(lambda x: aggregate(x, agg_dict, cat_treatment='majority')).reset_index(level=0, drop=True)

    # Ensure the columns are in the same order
    pd.testing.assert_frame_equal(result_df[result_df['group'] == 'A'].reset_index(drop=True), expected_df_group_A)
    pd.testing.assert_frame_equal(result_df[result_df['group'] == 'B'].reset_index(drop=True), expected_df_group_B)

def test_aggregate_with_groupby_proportions_as_columns():
    # Create a sample DataFrame
    data = {
        'group': pd.Categorical(['A', 'A', 'B', 'B']),
        'dry_mass': [1, 2, 3, 4],
        'assay1': [10, 20, 30, 40],
        'assay2': [100, 200, 300, 400],
        'other': [5, 6, 7, 8],
        'category': pd.Categorical(['a', 'a', 'b', 'a'])
    }
    df = pd.DataFrame(data)

    # Define the aggregation dictionary
    agg_dict = {
        'assay1': 'dry_mass',
        'assay2': 'dry_mass'
    }

    # Expected result for group 'A'
    expected_data_group_A = {
        'group_A': 1.0,
        'group_B': 0.0,
        'dry_mass': 3,
        'assay1': np.average([10, 20], weights=[1, 2]),
        'assay2': np.average([100, 200], weights=[1, 2]),
        'other': 11,
        'category_a': 1.0,  # Proportion of category 'a'
        'category_b': 0.0   # Proportion of category 'b'
    }
    expected_df_group_A = pd.DataFrame([expected_data_group_A])

    # Expected result for group 'B'
    expected_data_group_B = {
        'group_A': 0.0,
        'group_B': 1.0,
        'dry_mass': 7,
        'assay1': np.average([30, 40], weights=[3, 4]),
        'assay2': np.average([300, 400], weights=[3, 4]),
        'other': 15,
        'category_a': 0.5,  # Proportion of category 'a'
        'category_b': 0.5   # Proportion of category 'b'
    }
    expected_df_group_B = pd.DataFrame([expected_data_group_B])

    # Group by 'group' and apply the aggregate function
    result_df = df.groupby('group').apply(lambda x: aggregate(x, agg_dict, cat_treatment='proportions', proportions_as_columns=True)).reset_index(level=0, drop=True)

    # Ensure the columns are in the same order
    pd.testing.assert_frame_equal(result_df[result_df['group_A'] == 1.0].reset_index(drop=True), expected_df_group_A)
    pd.testing.assert_frame_equal(result_df[result_df['group_B'] == 1.0].reset_index(drop=True), expected_df_group_B)


def test_aggregate_grade_by_mass_and_density_by_volume():
    # Create a sample DataFrame
    data = {
        'mass': [1, 2, 3, 4],
        'volume': [0.5, 1.25, 1.6, 1.8],
        'grade_1': [0.1, 0.2, 0.3, 0.4],
        'grade_2': [0.2, 0.4, 0.6, 0.8],
        'density': list(np.array([1, 2, 3, 4]) / np.array([0.5, 1.25, 1.6, 1.8])),
        'other': [5, 6, 7, 8]
    }
    df = pd.DataFrame(data)

    # Define the aggregation dictionary
    agg_dict = {
        'grade_1': 'mass', 'grade_2': 'mass',
        'density': 'volume'
    }

    # Expected result
    expected_data = {
        'mass': 10,
        'volume': 5.15,
        'grade_1': np.average([0.1, 0.2, 0.3, 0.4], weights=[1, 2, 3, 4]),
        'grade_2': np.average([0.2, 0.4, 0.6, 0.8], weights=[1, 2, 3, 4]),
        'density': np.average(data['density'], weights=[0.5, 1.25, 1.6, 1.8]),
        'other': 26
    }
    expected_df = pd.DataFrame([expected_data])

    # Run the aggregate function
    result_df = aggregate(df, agg_dict)

    # Ensure the columns are in the same order
    pd.testing.assert_frame_equal(result_df, expected_df)

    # assert the average density is correct
    assert result_df['density'].values[0] == expected_data['mass'] / expected_data['volume']