from pathlib import Path

import yaml
from pandera import DataFrameSchema, Column, Check
from pandera.engines import pandas_engine
import pandas as pd
from pandera.io import _deserialize_check_stats


def load_schema_from_yaml(yaml_path: Path) -> DataFrameSchema:
    """Load a DataFrameSchema from a YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        schema_dict = yaml.safe_load(f)

    columns = {
        col_name: Column(**_deserialize_component_stats(col_stats))
        for col_name, col_stats in schema_dict["columns"].items()
    }

    return DataFrameSchema(
        columns=columns,
        checks=schema_dict.get("checks"),
        index=schema_dict.get("index"),
        dtype=schema_dict.get("dtype"),
        coerce=schema_dict.get("coerce", False),
        strict=schema_dict.get("strict", False),
        name=schema_dict.get("name", None),
        ordered=schema_dict.get("ordered", False),
        unique=schema_dict.get("unique", None),
        report_duplicates=schema_dict.get("report_duplicates", "all"),
        unique_column_names=schema_dict.get("unique_column_names", False),
        add_missing_columns=schema_dict.get("add_missing_columns", False),
        title=schema_dict.get("title", None),
        description=schema_dict.get("description", None),
    )


def _deserialize_component_stats(serialized_component_stats):
    dtype = serialized_component_stats.get("dtype")
    if dtype:
        dtype = pandas_engine.Engine.dtype(dtype)

    description = serialized_component_stats.get("description")
    title = serialized_component_stats.get("title")

    checks = serialized_component_stats.get("checks")
    if checks is not None:
        checks = [
            _deserialize_check_stats(
                getattr(Check, check_name), check_stats, dtype
            )
            for check_name, check_stats in checks.items()
        ]

    return {
        "title": title,
        "description": description,
        "dtype": dtype,
        "checks": checks,
        **{
            key: serialized_component_stats.get(key)
            for key in [
                "name",
                "nullable",
                "unique",
                "coerce",
                "required",
                "regex",
                "metadata"
            ]
            if key in serialized_component_stats
        },
    }
