from pathlib import Path

from omfpandas.base import OMFPandasBase
from omfpandas.blockmodel import blockmodel_to_parquet


class OMFDataConverter(OMFPandasBase):
    """A class to handle conversions between OMF and other formats."""

    def __init__(self, filepath: Path):
        """Instantiate the OMFConverter object

        Args:
            filepath: Path to the OMF file.
        """
        if not filepath.exists():
            raise FileNotFoundError(f'File does not exist: {filepath}')
        super().__init__(filepath)

    def blockmodel_to_parquet(self, blockmodel_name: str, parquet_filepath: Path,
                              with_geometry_index: bool = True, allow_overwrite: bool = False):
        """Write a VolumeElement to a Parquet file.

        Args:
            blockmodel_name (str): The name of the VolumeElement to convert.
            parquet_filepath (Path): The path to the Parquet file to write.
            with_geometry_index (bool): If True, includes geometry index in the DataFrame. Default is True.
            allow_overwrite (bool): If True, overwrite the existing Parquet file. Default is False.

        Raises:
            ValueError: If the element retrieved is not a VolumeElement.
        """
        bm = self.get_element_by_name(blockmodel_name)
        if bm.__class__.__name__ not in ['RegularBlockModel', 'TensorGridBlockModel']:
            raise ValueError(f"Element '{blockmodel_name}' is not a supported BlockModel in the OMF file: {self.filepath}")

        blockmodel_to_parquet(blockmodel=bm, out_path=parquet_filepath,
                              with_geometry_index=with_geometry_index, allow_overwrite=allow_overwrite)
