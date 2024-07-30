from pathlib import Path
from typing import Optional

import pandas as pd

from omfpandas.base import OMFPandasBase
from omfpandas.blockmodel import blockmodel_to_df


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

    def read_blockmodel(self, blockmodel_name: str, variables: Optional[list[str]] = None,
                        with_geometry_index: bool = True) -> pd.DataFrame:
        """Return a DataFrame from a BlockModel.

        Only variables assigned to the `cell` (as distinct from the grid `points`) are loaded.

        Args:
            blockmodel_name (str): The name of the BlockModel to convert.
            variables (Optional[list[str]]): The variables to include in the DataFrame. If None, all variables are
            included.
            with_geometry_index (bool): If True, includes geometry index in the DataFrame. Default is True.

        Returns:
            pd.DataFrame: The DataFrame representing the BlockModel.
        """
        bm = self.get_element_by_name(blockmodel_name)
        # check the element retrieved is the expected type
        if bm.__class__.__name__ not in ['RegularBlockModel', 'TensorGridBlockModel']:
            raise ValueError(f"Element '{bm}' is not a supported BlockModel in the OMF file: {self.filepath}")

        return blockmodel_to_df(bm, variables=variables, with_geometry_index=with_geometry_index)

