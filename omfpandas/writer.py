from pathlib import Path
from typing import Optional

import omf
import pandas as pd
from omf import OMFWriter

from omfpandas.base import OMFPandasBase
from omfpandas.volume import df_to_volume


class OMFPandasWriter(OMFPandasBase):
    """A class to write pandas dataframes to an OMF file.

    Attributes:
        filepath (Path): Path to the OMF file.
    """

    def __init__(self, filepath: Path):
        """Instantiate the OMFPandasWriter object.

        Args:
            filepath (Path): Path to the OMF file.
        """
        super().__init__(filepath)

        if not filepath.exists():
            # log a message and create a new project
            project = omf.Project()
            project.name = filepath.stem
            project.description = f"OMF file created by OMFPandasWriter: {filepath.name}"
            self._logger.info(f"Creating new OMF file: {filepath}")
            OMFWriter(project, str(filepath))

        super().__init__(filepath)

    def write_volume(self, blocks: pd.DataFrame, volume_name: str, allow_overwrite: bool = False):
        """Write a dataframe to a VolumeElement.

        Only dataframes with centroid (x, y, z) and block dims (dx, dy, dz) indexes are supported.

        Args:
            blocks (pd.DataFrame): The dataframe to write to the VolumeElement.
            volume_name (str): The name of the VolumeElement to write to.
            allow_overwrite (bool): If True, overwrite the existing VolumeElement. Default is False.

        Raises:
            ValueError: If the element retrieved is not a VolumeElement.
        """

        # if self.get_element_by_name(volume_name) is not None and not allow_overwrite:
        #     raise ValueError(f"VolumeElement '{volume_name}' already exists in the OMF file: {self.omf_filepath}.  "
        #                      f"If you want to overwrite, set allow_overwrite=True.")
        volume = df_to_volume(blocks, volume_name)
        if volume.name in [element.name for element in self.project.elements]:
            if not allow_overwrite:
                raise ValueError(f"VolumeElement '{volume_name}' already exists in the OMF file: {self.filepath}.  "
                                 f"If you want to overwrite, set allow_overwrite=True.")
            else:
                # remove the existing volume from the project
                volume_to_remove = [element for element in self.project.elements if element.name == volume.name][0]
                self.project.elements.remove(volume_to_remove)

        self.project.elements.append(volume)
        # write the file
        OMFWriter(project=self.project, fname=str(self.filepath))
