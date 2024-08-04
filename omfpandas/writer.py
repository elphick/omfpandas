import getpass
from pathlib import Path
from typing import Optional

import omf
import pandas as pd

import omfpandas
from omfpandas import OMFPandasReader
from omfpandas.base import OMFPandasBase
from omfpandas.blockmodel import df_to_blockmodel, series_to_attribute


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
        self.user_id = getpass.getuser()

        if not filepath.exists():
            # log a message and create a new project
            project = omf.Project()
            project.name = filepath.stem
            project.description = f"OMF file created by OMFPandasWriter: {filepath.name}"
            self._logger.info(f"Creating new OMF file: {filepath}")
            omf.save(project, str(filepath))

        super().__init__(filepath)

    def write_blockmodel(self, blocks: pd.DataFrame, blockmodel_name: str, allow_overwrite: bool = False):
        """Write a dataframe to a BlockModel.

        Only dataframes with centroid (x, y, z) and block dims (dx, dy, dz) indexes are supported.

        Args:
            blocks (pd.DataFrame): The dataframe to write to the BlockModel.
            blockmodel_name (str): The name of the BlockModel to write to.
            allow_overwrite (bool): If True, overwrite the existing BlockModel. Default is False.

        Raises:
            ValueError: If the element retrieved is not a BlockModel.
        """

        bm = df_to_blockmodel(blocks, blockmodel_name)
        if bm.name in [element.name for element in self.project.elements]:
            if not allow_overwrite:
                raise ValueError(f"BlockModel '{blockmodel_name}' already exists in the OMF file: {self.filepath}.  "
                                 f"If you want to overwrite, set allow_overwrite=True.")
            else:
                # remove the existing volume from the project
                volume_to_remove = [element for element in self.project.elements if element.name == bm.name][0]
                self.project.elements.remove(volume_to_remove)

        self.project.elements.append(bm)
        # write the file
        omf.save(project=self.project, filename=str(self.filepath), mode='w')

    def write_blockmodel_attribute(self, blockmodel_name: str, attribute_name: str, data: pd.Series,
                                   allow_overwrite: bool = False):
        """Write data to a specific attribute of a BlockModel.

        Args:
            blockmodel_name (str): The name of the BlockModel.
            attribute_name (str): The name of the attribute.
            data (pd.Series): The data to write to the attribute.
            allow_overwrite (bool): If True, overwrite the existing attribute. Default is False.
        """
        bm = self.get_element_by_name(blockmodel_name)
        attrs: list[str] = self.get_element_attribute_names(bm)
        if attribute_name in attrs:
            if allow_overwrite:
                bm.attributes[attribute_name] = series_to_attribute(data)
            else:
                raise ValueError(f"Attribute '{attribute_name}' already exists in BlockModel '{blockmodel_name}'.  "
                                 f"If you want to overwrite, set allow_overwrite=True.")
        else:
            bm.attributes[attribute_name] = series_to_attribute(data)

        self._delete_profile_report(blockmodel_name)

        # Save the changes
        omf.save(project=self.project, filename=str(self.filepath), mode='w')

    def delete_blockmodel_attribute(self, blockmodel_name: str, attribute_name: str):
        """Delete an attribute from a BlockModel.

        Args:
            blockmodel_name (str): The name of the BlockModel.
            attribute_name (str): The name of the attribute.
        """
        bm = self.get_element_by_name(blockmodel_name)
        attrs: list[str] = self.get_element_attribute_names(bm)
        if attribute_name in attrs:
            del bm.attributes[attribute_name]
        else:
            raise ValueError(f"Attribute '{attribute_name}' not found in BlockModel '{blockmodel_name}'.")

        self._delete_profile_report(blockmodel_name)

        # Save the changes
        omf.save(project=self.project, filename=str(self.filepath), mode='w')

    def profile_blockmodel(self, blockmodel_name: str, query: Optional[str] = None) -> 'ProfileReport':
        """Profile a BlockModel.

        Profiling will be skipped if the data has not changed.

        Args:
            blockmodel_name (str): The name of the BlockModel to profile.
            query (Optional[str]): A query to filter the data before profiling.

        Returns:
            pd.DataFrame: The profiled data.
        """
        try:
            from ydata_profiling import ProfileReport
        except ImportError:
            raise ImportError("ydata_profiling is required to run this method.  "
                              "Please install it by running 'poetry install omfpandas --extras profile' "
                              "or 'pip install ydata_profiling'")
        df: pd.DataFrame = OMFPandasReader(self.filepath).read_blockmodel(blockmodel_name, query=query)
        el = self.get_element_by_name(blockmodel_name)
        bm_type = str(type(el)).split('.')[-1].rstrip("'>")
        dataset: dict = {"description": f"{el.description} Filter: {query if query else 'no_filter'}",
                         "creator": self.user_id}

        profile = ProfileReport(df, title=f"{el.name} {bm_type}", dataset=dataset)

        # persist the profile report as json and html to the omf file
        d_profile: dict = {query if query else 'no_filter': {'json': profile.to_json(), 'html': profile.to_html()}}
        if el.metadata.get('profile'):
            el.metadata['profile'] = {**el.metadata['profile'], **d_profile}
        else:
            el.metadata['profile'] = d_profile

        omf.save(project=self.project, filename=str(self.filepath), mode='w')

        return profile

    def _delete_profile_report(self, blockmodel_name: str):
        """Delete the profile report from the OMF file when data has changed."""
        bm = self.get_element_by_name(blockmodel_name)

        if 'profile' in bm.metadata:
            del bm.metadata['profile']