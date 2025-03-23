import numpy as np
import pandas as pd
import pandera as pa
from pandera.api.pandas.components import Column
from omfpandas.utils.pandera_utils import DataFrameMetaProcessor


def test_rename_from_meta_alias():
    schema = pa.DataFrameSchema({
        "original_name": Column(
            dtype=str,
            metadata={"alias": "alias_name"}
        )
    })
    processor = DataFrameMetaProcessor(schema)
    df = pd.DataFrame({"alias_name": ["value1", "value2"]})
    df = processor.rename_from_meta_alias(df)
    assert "original_name" in df.columns
    assert "alias_name" not in df.columns
    assert df["original_name"].tolist() == ["value1", "value2"]


def test_calculate_from_meta_formula():
    schema = pa.DataFrameSchema({
        "calculated_column": Column(
            dtype='Float32',
            coerce=True,
            metadata={"calculation": "col1 + col2"}
        )
    })
    processor = DataFrameMetaProcessor(schema)
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    df = processor.calculate_from_meta_calculation(df)
    assert "calculated_column" in df.columns
    assert df["calculated_column"].tolist() == [4, 6]


def test_preprocess_and_validate():
    schema = pa.DataFrameSchema({
        "original_name": Column(
            dtype=str,
            metadata={"alias": "alias_name"}
        ),
        "calculated_column": Column(
            dtype='Float32',
            coerce=True,
            metadata={"calculation": "col1 + col2"}
        )
    })
    processor = DataFrameMetaProcessor(schema)
    df = pd.DataFrame({"alias_name": ["value1", "value2"], "col1": [1, 2], "col2": [3, 4]})
    df = processor.preprocess(df)
    df = processor.validate(df)
    assert "original_name" in df.columns
    assert "calculated_column" in df.columns
    assert df["original_name"].tolist() == ["value1", "value2"]
    assert df["calculated_column"].tolist() == [4, 6]


def test_round_to_decimals():
    schema = pa.DataFrameSchema({
        "rounded_column": Column(
            dtype=float,
            coerce=True,
            metadata={"decimals": 2}
        )
    })
    processor = DataFrameMetaProcessor(schema)
    df = pd.DataFrame({"rounded_column": [1.234, 2.345, 3.456]})
    df = processor.round_to_decimals(df)
    assert df["rounded_column"].tolist() == [1.23, 2.35, 3.46]
