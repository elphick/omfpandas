# omfpandas

A pandas (and parquet) interface for the [Open Mining Format package (omf)](https://omf.readthedocs.io/en/latest/).

When working with OMF files, it is often useful to convert the data to a pandas DataFrame.
This package provides a simple interface to do so.

The parquet format is a nice, compact efficient format to persist pandas DataFrames.
This package also provides a simple interface to convert an omf element to a parquet file.
When datasets do not fit into memory, parquet files can be read in chunks or by column.

## Installation

```bash
pip install omfpandas
```

## Roadmap

- [ ] 0.2.0 - Add support for reading a VolumeElement (Block Model) from an OMF file as a pandas DataFrame. 
  Export a VolumeElement as a parquet file.
- [ ] 0.3.0 - Add support for writing a DataFrame to an OMF VolumeElement
- [ ] 0.4.0 - Add support for low-memory/out-of-core writing an omf element to parquet
- [ ] ...
