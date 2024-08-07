from pathlib import Path
from typing import Optional

import pandas as pd

from omfpandas.base import OMFPandasBase
from omfpandas.blockmodel import blockmodel_to_df, create_index


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

    def read_blockmodel(self, blockmodel_name: str, attributes: Optional[list[str]] = None,
                        query: Optional[str] = None) -> pd.DataFrame:
        """Return a DataFrame from a BlockModel.

        Only variables assigned to the `cell` (as distinct from the grid `points`) are loaded.

        Args:
            blockmodel_name (str): The name of the BlockModel to convert.
            attributes (Optional[list[str]]): The attributes/variables to include in the DataFrame. If None, all
            variables are included.
            query (Optional[str]): A query string to filter the DataFrame. Default is None.

        Returns:
            pd.DataFrame: The DataFrame representing the BlockModel.
        """
        bm = self.get_element_by_name(blockmodel_name)
        # check the element retrieved is the expected type
        if bm.__class__.__name__ not in ['RegularBlockModel', 'TensorGridBlockModel']:
            raise ValueError(f"Element '{bm}' is not a supported BlockModel in the OMF file: {self.filepath}")

        return blockmodel_to_df(bm, variables=attributes, query=query)

    def read_block_models(self, blockmodel_attributes: dict[str, list[str]]) -> pd.DataFrame:
        """Return a DataFrame from multiple BlockModels.

        Args:
            blockmodel_attributes (dict[str, list[str]]): A dictionary of BlockModel names and the variables to include.
            If the dict value is None, all attributes in the blockmodel (key) are included.


        Returns:
            pd.DataFrame: The DataFrame representing the merged BlockModels.
        """
        block_models: dict[str, pd.DataFrame] = {}
        geometry_indexes: dict[str, pd.MultiIndex] = {}
        for bm_name, requested_attrs in blockmodel_attributes.items():
            # check that the requested attrs exist in the specified bm
            available_attrs = self.element_attributes[bm_name]
            if requested_attrs is None:
                requested_attrs = available_attrs
            else:
                missing_attrs = set(requested_attrs) - set(available_attrs)
                if missing_attrs:
                    raise ValueError(f"Attributes {missing_attrs} not found in BlockModel '{bm_name}'. "
                                     f"Available attributes are: {available_attrs}")

            block_models[bm_name] = self.read_blockmodel(blockmodel_name=bm_name, attributes=requested_attrs)
            geometry_indexes[bm_name] = create_index(blockmodel=self.get_element_by_name(bm_name))

        # validate the indexes are equivalent
        def ensure_identical_indexes(index_dict: dict[str, pd.MultiIndex]) -> None:
            if not index_dict:
                return

            first_index = next(iter(index_dict.values()))
            for name, index in index_dict.items():
                if not first_index.equals(index):
                    raise ValueError(f"Index for '{name}' is different from the first index.")

        ensure_identical_indexes(geometry_indexes)

        return pd.concat(block_models.values(), axis=1)
