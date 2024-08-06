import getpass
import json
from pathlib import Path
from typing import Optional

import omf
import pandas as pd
import ydata_profiling

from omfpandas import OMFPandasReader
from omfpandas.base import OMFPandasBase
from omfpandas.blockmodel import df_to_blockmodel, series_to_attribute

from omfpandas.extras import _import_ydata_profiling, _import_pandera
from omfpandas.utils.timer import log_timer


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

    @log_timer()
    def write_blockmodel(self, blocks: pd.DataFrame, blockmodel_name: str, pd_schema_filepath: Optional[Path] = None,
                         allow_overwrite: bool = False):
        """Write a dataframe to a BlockModel.

        Only dataframes with centroid (x, y, z) and block dims (dx, dy, dz) indexes are supported.

        Args:
            blocks (pd.DataFrame): The dataframe to write to the BlockModel.
            blockmodel_name (str): The name of the BlockModel to write to.
            pd_schema_filepath (Optional[Path]): The path to the Pandera schema file. Default is None.  If provided,
            the schema will be used to validate the dataframe before writing.
            allow_overwrite (bool): If True, overwrite the existing BlockModel. Default is False.

        Raises:
            ValueError: If the element retrieved is not a BlockModel.
        """

        if pd_schema_filepath:
            pa = _import_pandera()
            # validate the dataframe, which may modify it via coercion
            pd_schema = pa.DataFrameSchema.from_yaml(pd_schema_filepath)
            self._logger.info(f"Validating dataframe with schema: {pd_schema_filepath}")
            blocks = pd_schema.validate(blocks)
            self._logger.info(f"Writing dataframe to BlockModel: {blockmodel_name}")
            bm = df_to_blockmodel(blocks, blockmodel_name)
            # persist the schema inside the omf file
            bm.metadata['pd_schema'] = pd_schema.to_json()
        else:
            self._logger.info(f"Writing dataframe to BlockModel: {blockmodel_name}")
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

    def write_blockmodel_attribute(self, blockmodel_name: str, series: pd.Series,
                                   allow_overwrite: bool = False):
        """Write data to a specific attribute of a BlockModel.

        Args:
            blockmodel_name (str): The name of the BlockModel.
            series (pd.Series): The data to write to the attribute.
            allow_overwrite (bool): If True, overwrite the existing attribute. Default is False.
        """

        bm = self.get_element_by_name(blockmodel_name)
        if bm.metadata.get('pd_schema'):
            pa = _import_pandera()
            # validate the data
            schema = pa.io.from_json(bm.metadata['pd_schema'])
            series = schema.validate(series.to_frame())
            series = series.iloc[:, 0]  # back to series

        attrs: list[str] = self.get_element_attribute_names(blockmodel_name)
        if series.name in attrs:
            if allow_overwrite:
                # get the index in the list
                attr_pos = attrs.index(str(series.name))
                bm.attributes[attr_pos] = series_to_attribute(series)
            else:
                raise ValueError(f"Attribute '{series.name}' already exists in BlockModel '{blockmodel_name}'.  "
                                 f"If you want to overwrite, set allow_overwrite=True.")
        else:
            bm.attributes.append(series_to_attribute(series))

        self._delete_profile_report(blockmodel_name)

        # todo: re-profile...

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

    @log_timer()
    def profile_blockmodel(self, blockmodel_name: str, query: Optional[str] = None):
        """Profile a BlockModel.

        Profiling will be skipped if the data has not changed.

        Args:
            blockmodel_name (str): The name of the BlockModel to profile.
            query (Optional[str]): A query to filter the data before profiling.

        Returns:
            pd.DataFrame: The profiled data.
        """

        _import_ydata_profiling()

        df: pd.DataFrame = OMFPandasReader(self.filepath).read_blockmodel(blockmodel_name, query=query)
        el = self.get_element_by_name(blockmodel_name)
        bm_type = str(type(el)).split('.')[-1].rstrip("'>")
        dataset: dict = {"description": f"{el.description} Filter: {query if query else 'no_filter'}",
                         "creator": self.user_id, "url": self.filepath.as_uri()}
        column_descriptions: dict = {}
        if el.metadata.get('pd_schema'):
            column_defs: dict = json.loads(el.metadata['pd_schema'])['columns']
            column_descriptions = {k: f"{v['title']}: {v['description']}" for k, v in column_defs.items()}

        profile: ydata_profiling.ProfileReport = df.profile_report(title=f"{el.name} {bm_type}", dataset=dataset,
                                                                   variables={"descriptions": column_descriptions})

        # persist the profile report as html to the omf file, larger but cannot serialise the ProfileReport object,
        # nor recreate the report from json.
        d_profile: dict = {query if query else 'no_filter': profile.to_html()}

        if el.metadata.get('profile'):
            el.metadata['profile'] = {**el.metadata['profile'], **d_profile}
        else:
            el.metadata['profile'] = d_profile

        omf.save(project=self.project, filename=str(self.filepath), mode='w')

        return profile

    def write_block_model_schema(self, blockmodel_name: str, pd_schema_filepath: Path):
        """Write a Pandera schema to the OMF file.

        Args:
            blockmodel_name (str): The name of the BlockModel.
            pd_schema_filepath (Path): The path to the Pandera schema yaml file.
        """
        pa = _import_pandera()
        bm = self.get_element_by_name(blockmodel_name)
        pd_schema = pa.DataFrameSchema.from_yaml(pd_schema_filepath)
        bm.metadata['pd_schema'] = pd_schema.to_json()

        el = self.get_element_by_name(blockmodel_name)
        schema_title = pd_schema.title if pd_schema.title else ''
        schema_description = pd_schema.description if pd_schema.description else ''
        el.description = f"{schema_title}: {schema_description}"

        omf.save(project=self.project, filename=str(self.filepath), mode='w')

    def _delete_profile_report(self, blockmodel_name: str):
        """Delete the profile report from the OMF file when data has changed."""
        bm = self.get_element_by_name(blockmodel_name)

        if 'profile' in bm.metadata:
            del bm.metadata['profile']
