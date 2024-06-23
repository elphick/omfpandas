from pathlib import Path
from typing import Optional

import pandas as pd
from omf import OMFReader

from omfpandas.base import OMFPandasBase
from omfpandas.volume import volume_to_df, volume_to_parquet


class OMFPandasReader(OMFPandasBase):
    """A class to read an OMF file to a pandas DataFrame.

    Attributes:
        filepath (Path): Path to the OMF file.

    """
    def __init__(self, filepath: Path):
        """Instantiate the OMFPandasReader object

        Args:
            filepath: Path to the OMF file.
        """
        if not filepath.exists():
            raise FileNotFoundError(f'File does not exist: {filepath}')
        super().__init__(filepath)

    def read_volume(self, volume_name: str, variables: Optional[list[str]] = None,
                    with_geometry_index: bool = True) -> pd.DataFrame:
        """Return a DataFrame from a VolumeElement.

        Only variables assigned to the `cell` (as distinct from the grid `points`) are loaded.

        Args:
            volume_name (str): The name of the VolumeElement to convert.
            variables (Optional[list[str]]): The variables to include in the DataFrame. If None, all variables are included.
            with_geometry_index (bool): If True, includes geometry index in the DataFrame. Default is True.

        Returns:
            pd.DataFrame: The DataFrame representing the VolumeElement.
        """
        volume = self.get_element_by_name(volume_name)
        # check the element retrieved is the expected type
        if volume.__class__.__name__ != 'VolumeElement':
            raise ValueError(f"Element '{volume}' is not a VolumeElement in the OMF file: {self.omf_filepath}")

        return volume_to_df(volume, variables=variables, with_geometry_index=with_geometry_index)

