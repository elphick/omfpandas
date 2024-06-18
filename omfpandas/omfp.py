from pathlib import Path
from typing import Optional

import pandas as pd
from omf import OMFReader

from omfpandas.volume import volume_to_df, volume_to_parquet


class OMFPandas:
    def __init__(self, omf_filepath: Path):
        """Instantiate the OMFPandas object.

        :param omf_filepath: Path to the OMF file.
        """
        if not omf_filepath.exists():
            raise FileNotFoundError(f'File not found: {omf_filepath}')
        elif not omf_filepath.suffix == '.omf':
            raise ValueError(f'File is not an OMF file: {omf_filepath}')
        self.omf_filepath = omf_filepath
        self._elements = OMFReader(str(omf_filepath)).get_project().elements
        self.elements: dict[str, str] = {e.name: e.subtype for e in self._elements}

    def get_element_by_name(self, element_name: str):
        """Get an element by its name.

        :param element_name: The name of the element to retrieve.
        :return:
        """
        element = [e for e in self._elements if e.name == element_name]
        if not element:
            raise ValueError(f"Element '{element_name}' not found in the OMF file: {self.omf_filepath.name}. "
                             f"Available elements are: {list(self.elements.keys())}")
        elif len(element) > 1:
            raise ValueError(f"Multiple elements with the name '{element_name}' found in the OMF file: "
                             f"{self.omf_filepath.name}")
        return element[0]

    def volume_to_df(self, volume_name: str, variables: Optional[list[str]] = None,
                     with_geometry_index: bool = True) -> pd.DataFrame:
        """Return a DataFrame from a VolumeElement.

        Only variables assigned to the `cell` (as distinct from the grid `points`) are loaded.

        :param volume_name: The name of the VolumeElement to convert.
        :param variables: The variables to include in the DataFrame.  If None, all variables are included.
        :param with_geometry_index:
        :return: The DataFrame representing the VolumeElement.
        """
        volume = self.get_element_by_name(volume_name)
        # check the element retrieved is the expected type
        if volume.__class__.__name__ != 'VolumeElement':
            raise ValueError(f"Element '{volume}' is not a VolumeElement in the OMF file: {self.omf_filepath}")

        return volume_to_df(volume, variables=variables, with_geometry_index=with_geometry_index)

    def volume_to_parquet(self, volume_name: str, parquet_filepath: Path):
        """Write a VolumeElement to a Parquet file.

        :param volume_name: The name of the VolumeElement to convert.
        :param parquet_filepath: The path to the Parquet file to write.
        :return:
        """
        volume = self.get_element_by_name(volume_name)
        # check the element retrieved is the expected type
        if volume.__class__.__name__ != 'VolumeElement':
            raise ValueError(f"Element '{volume}' is not a VolumeElement in the OMF file: {self.omf_filepath}")

        volume_to_parquet(volume, parquet_filepath)
