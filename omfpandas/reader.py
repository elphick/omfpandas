import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from omfpandas.base import OMFPandas, SUPPORTED_BM_TYPES
from omfpandas.blockmodel import OMFBlockModel
from omfpandas.blockmodels import multiindex_to_encoded_index
from omfpandas.blockmodels.convert_blockmodel import blockmodel_to_df
from omfpandas.blockmodels.geometry import Geometry
from omfpandas.utils.pandas_utils import parse_vars_from_expr

PathLike = Union[str, Path, os.PathLike]


class OMFPandasReader(OMFPandas):
    """A class to read an OMF file to a pandas DataFrame.

    Attributes:
        filepath (Path): Path to the OMF file.

    """

    def __init__(self, filepath: PathLike):
        """Instantiate the OMFPandasReader object

        Args:
            filepath: Path to the OMF file.
        """
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File does not exist: {filepath}")
        super().__init__(filepath)

    def read_blockmodel(
            self,
            blockmodel_name: str,
            attributes: Optional[list[str]] = None,
            query: Optional[str] = None,
            index_filter: Optional[list[int]] = None,
            encode_index: bool = False,
    ) -> pd.DataFrame:
        """Return a DataFrame from a BlockModel.

        Only variables assigned to the `cell` (as distinct from the grid `points`) are loaded.

        Args:
            blockmodel_name (str): The name of the BlockModel to read. Use dot notation for composite (e.g., Composite.BlockModel).
            attributes (Optional[list[str]]): The attributes/variables to include in the DataFrame. If None, all
                variables are included.
            query (Optional[str]): A query string to filter the DataFrame. Default is None.
            index_filter (Optional[list[int]]): A list of indexes to filter the DataFrame. Default is None.
            encode_index (bool): If True, encode the index to a single integer.

        Returns:
            pd.DataFrame: The DataFrame representing the BlockModel.
        """
        bm = self.get_element_by_name(blockmodel_name)
        # check the element retrieved is the expected type
        if bm.__class__.__name__ not in ["RegularBlockModel", "TensorGridBlockModel"]:
            raise ValueError(
                f"Element '{bm}' is not a supported BlockModel in the OMF file: {self.filepath}"
            )
        res: pd.DataFrame = blockmodel_to_df(
            bm, variables=attributes, query=query, index_filter=index_filter
        )
        if encode_index:
            res.index = multiindex_to_encoded_index(res.index)
        return res

    def read_block_models(
            self,
            blockmodel_attributes: dict[str, list[str]],
            query: Optional[str] = None,
            encode_index: bool = False,
    ) -> pd.DataFrame:
        """Return a DataFrame from multiple BlockModels.

        Args:
            blockmodel_attributes (dict[str, list[str]]): A dictionary of BlockModel names and the variables to include.
                If the dict value is None, all attributes in the blockmodel (key) are included.
            query (Optional[str]): A query string to filter the DataFrame. Default is None.
            encode_index: If True, encode the index to a single integer.

        Returns:
            pd.DataFrame: The DataFrame representing the merged BlockModels.
        """
        block_models: dict[str, pd.DataFrame] = {}
        geometry_indexes: dict[str, pd.MultiIndex] = {}
        index_filter = None
        if query:
            # we use the query to generate an index_filter to filter the block models since they may not have
            # the query attributes available in the given model.
            query_attrs: list[str] = parse_vars_from_expr(query)
            # extract the attributes for the query_attributes, noting that they may be in different block models
            chunks: list[pd.Series] = []
            for bm_name in blockmodel_attributes:
                for attr in query_attrs:
                    if attr in self.blockmodel_attributes[bm_name]:
                        chunks.append(
                            self.read_blockmodel(blockmodel_name=bm_name,
                                                 attributes=[attr], encode_index=encode_index)[attr])
            tmp_df: pd.DataFrame = pd.concat(chunks, axis=1).query(query)
            # get the index locations of the filtered index relative to the full index
            filtered_index = tmp_df.index
            # full_index = geometry_to_index(self.get_element_by_name(self._elements[0].name).geometry)
            full_index: pd.MultiIndex = self.get_bm_geometry(
                blockmodel_name=list(blockmodel_attributes.keys())[0]
            ).to_multi_index()
            # Find the intersection of the two MultiIndex objects
            intersection = full_index.intersection(filtered_index)
            index_filter = full_index.get_indexer(intersection)

        for bm_name, requested_attrs in blockmodel_attributes.items():
            # check that the requested attrs exist in the specified bm
            available_attrs = self.blockmodel_attributes[bm_name]
            if requested_attrs is None:
                requested_attrs = available_attrs
            else:
                missing_attrs = set(requested_attrs) - set(available_attrs)
                if missing_attrs:
                    raise ValueError(
                        f"Attributes {missing_attrs} not found in BlockModel '{bm_name}'. "
                        f"Available attributes are: {available_attrs}"
                    )

            block_models[bm_name] = self.read_blockmodel(
                blockmodel_name=bm_name,
                attributes=requested_attrs,
                index_filter=index_filter if query else None,
            )
            # geometry_indexes[bm_name] = geometry_to_index(self.get_bm_geometry(bm_name))
            geometry_indexes[bm_name] = self.get_bm_geometry(
                blockmodel_name=bm_name
            ).to_multi_index()

        # validate the indexes are equivalent
        def ensure_identical_indexes(index_dict: dict[str, pd.MultiIndex]) -> None:
            if not index_dict:
                return

            first_index = next(iter(index_dict.values()))
            for name, index in index_dict.items():
                if not first_index.equals(index):
                    raise ValueError(
                        f"Index for '{name}' is different from the first index."
                    )

        ensure_identical_indexes(geometry_indexes)

        return pd.concat(block_models.values(), axis=1)

    def plot_blockmodel(
            self,
            blockmodel_name: str,
            scalar: str,
            threshold: bool = True,
            show_edges: bool = True,
            show_axes: bool = True,
    ) -> "pv.Plotter":
        """Plot the BlockModel using PyVista.

        Args:
            blockmodel_name (str): The name of the BlockModel to plot.
            scalar (str): The scalar to plot.
            threshold (bool): If True, plot the thresholded mesh. Default is True.
            show_edges (bool): If True, show the edges. Default is True.
            show_axes (bool): If True, show the axes. Default is True.

        Returns:
            pv.Plotter: The PyVista plotter object.
        """
        block_model = OMFBlockModel(self.get_element_by_name(blockmodel_name))
        return block_model.plot(
            scalar=scalar,
            threshold=threshold,
            show_edges=show_edges,
            show_axes=show_axes,
        )

    def find_nearest_centroid(
            self, x: float, y: float, z: float, blockmodel_name: Optional[str] = None
    ) -> tuple[float, float, float]:
        """Find the nearest centroid for provided x, y, z points using a math rounding approach considering the reference centroid.

        Args:
            x (float): X coordinate.
            y (float): Y coordinate.
            z (float): Z coordinate.
            blockmodel_name: The optional block model name.  If not provided, the geometry for the first Tensor
             block model is used.

        Returns:
            Tuple[float, float, float]: The coordinates of the nearest centroid.
        """

        # get the geometry for the first block model if not provided
        if blockmodel_name is None:
            blockmodel_names = [
                element_name
                for element_name, element_type in self.element_types.items()
                if element_type in SUPPORTED_BM_TYPES
            ]
            if not blockmodel_names:
                raise ValueError("No BlockModel found in the OMF file.")
            blockmodel_name = blockmodel_names[0]

        # get the geometry for the block model
        geometry: Geometry = OMFBlockModel(
            self.get_element_by_name(blockmodel_name)
        ).geometry
        # perform the lookup
        nearest_x, nearest_y, nearest_z = geometry.nearest_centroid_lookup(x, y, z)

        return nearest_x, nearest_y, nearest_z
