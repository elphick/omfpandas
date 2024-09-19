import getpass
import json
from pathlib import Path
from typing import Optional, Literal, Union

import numpy as np
import omf
import pandas as pd
import ydata_profiling

from omfpandas import OMFPandasReader
from omfpandas.audit import ChangeMessage
from omfpandas.base import OMFPandasBase
from omfpandas.blockmodel import df_to_blockmodel, series_to_attribute

from omfpandas.extras import _import_ydata_profiling, _import_pandera, _import_pandera_io
from omfpandas.utils.pandas import parse_vars_from_expr
from omfpandas.utils.timer import log_timer


class OMFPandasWriter(OMFPandasReader):
    """A class to write pandas dataframes to an OMF file.

    Attributes:
        filepath (Path): Path to the OMF file.
    """

    def __init__(self, filepath: Path):
        """Instantiate the OMFPandasWriter object.

        Args:
            filepath (Path): Path to the OMF file.
        """
        OMFPandasBase.__init__(self, filepath)
        self.user_id = getpass.getuser()

        if not filepath.exists():
            # log a message and create a new project
            project = omf.Project()
            project.name = filepath.stem
            project.description = f"OMF file created by OMFPandasWriter: {filepath.name}"
            self._logger.info(f"Creating new OMF file: {filepath}")
            self.project = project  # to enable the write_to_changelog method
            # create the audit record, which also saves the file
            self.write_to_changelog(element='None', action='create', description=f"File created: {filepath}")

        super().__init__(filepath)

    @log_timer()
    def write_blockmodel(self, blocks: pd.DataFrame, blockmodel_name: str,
                         pd_schema: Optional[Union[Path, dict]] = None,
                         allow_overwrite: bool = False):
        """Write a dataframe to a BlockModel.

        Only dataframes with centroid (x, y, z) and block dims (dx, dy, dz) indexes are supported.

        Args:
            blocks (pd.DataFrame): The dataframe to write to the BlockModel.
            blockmodel_name (str): The name of the BlockModel to write to.
            pd_schema (Optional[Union[Path, dict]]): The path to the Pandera schema file or a dict of the schema.
             efault is None.  If provided, the schema will be used to validate the dataframe before writing.
            allow_overwrite (bool): If True, overwrite the existing BlockModel. Default is False.

        Raises:
            ValueError: If the element retrieved is not a BlockModel.
        """

        if pd_schema is not None:
            pa = _import_pandera()
            if not isinstance(pd_schema, (Path, dict)):
                raise ValueError("pd_schema must be a Path to a Pandera schema file or a dict of the schema.")
            elif isinstance(pd_schema, Path) and not pd_schema.exists():
                raise FileNotFoundError(f"Schema file not found: {pd_schema}")
            elif isinstance(pd_schema, dict):
                paio = _import_pandera_io()
                pd_schema = paio.deserialize_schema(pd_schema)
            elif isinstance(pd_schema, Path):
                pd_schema = pa.DataFrameSchema.from_yaml(pd_schema)
                self._logger.info(f"Validating dataframe with schema: {pd_schema}")

            # validate the dataframe, which may modify it via coercion
            blocks = pd_schema.validate(blocks)
            self._logger.info(f"Writing dataframe to BlockModel: {blockmodel_name}")
            bm = df_to_blockmodel(blocks, blockmodel_name)
            # persist the schema inside the omf file
            bm.description = pd_schema.description
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

        # create the audit record, which also saves the file
        self.write_to_changelog(element=bm.name, action='create', description='BlockModel written')

    def add_calculated_blockmodel_attributes(self, blockmodel_name: str, calc_definitions: dict[str, str]):
        """Add a calculated attribute to a BlockModel.

        Calculated attributes reduce storage space by storing the calculation expression instead of the data.
        When the attribute is accessed, the expression is evaluated and the result returned.
        The calculation expression must be a valid pandas expression, and is stored in the metadata of the
        blockmodel object.

        Args:
            blockmodel_name (str): The name of the BlockModel.
            calc_definitions (dict[str, str]): A dictionary of attribute names and calculation expressions.
        """
        bm = self.get_element_by_name(blockmodel_name)
        # confirm the element is a BlockModel
        if bm.__class__.__name__ not in ['RegularBlockModel', 'TensorGridBlockModel']:
            raise ValueError(f"Element '{bm}' is not a supported BlockModel in the OMF file: {self.filepath}")
        for attr_name, expr in calc_definitions.items():
            # check the attribute does not already exist
            if attr_name in self.get_element_attribute_names(blockmodel_name):
                raise ValueError(f"Attribute '{attr_name}' already exists in BlockModel '{blockmodel_name}'.")
            attrs_in_scope = list(set(parse_vars_from_expr(expr)))
            # validate that the attributes in the expression are in the schema
            for attr in attrs_in_scope:
                if attr not in self.get_element_attribute_names(blockmodel_name):
                    raise ValueError(f"Expression attribute '{attr}' not found in BlockModel '{blockmodel_name}'.")

            # Load the head dataset containing the attributes in the expression to validate the expression is valid,
            # though if the attribute exists in the schema file then it will be validated on the entire dataset.
            index_filter = list(range(0, 6))
            if bm.metadata.get('pd_schema'):
                pa = _import_pandera()
                col_schema = pa.io.from_json(bm.metadata['pd_schema']).get(attr_name)
                if col_schema:
                    index_filter = None

            # validate the expression
            df: pd.DataFrame = self.read_blockmodel(blockmodel_name, attributes=attrs_in_scope,
                                                    index_filter=index_filter)
            try:
                df.eval(expr)
            except Exception as e:
                raise ValueError(f"Expression '{expr}' failed during evaluation: {e}")

            # persist the configuration to the blockmodel metadata
            if 'calculated_attributes' in bm.metadata:
                bm.metadata['calculated_attributes'][attr_name] = expr
            else:
                bm.metadata['calculated_attributes'] = {attr_name: expr}

            self.write_to_changelog(element=blockmodel_name, action='create',
                                    description=f"Calculated attribute [{attr_name}] added")

    def write_to_changelog(self, element: str, action: Literal['create', 'update', 'delete'], description: str):
        """Write a change message to the OMF file.

        Args:
            element: The name of the element that was changed
            action: The action taken on the object
            description: Description of the change

        Returns:

        """

        if 'changelog' not in self.project.metadata:
            self.project.metadata['changelog'] = []
        msg = ChangeMessage(element=element, user=self.user_id, action=action, description=description)
        self.project.metadata['changelog'].append(str(msg))
        omf.save(project=self.project, filename=str(self.filepath), mode='w')
        self.project = omf.load(str(self.filepath))

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

        self.write_to_changelog(element=bm.name, action='create', description=f"Attribute [{series.name}] written")

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

        # create the audit record, which also saves the file
        self.write_to_changelog(element=bm.name, action='delete', description=f"{attribute_name} deleted")

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

        self.write_to_changelog(element=blockmodel_name, action='create',
                                description=f"Profiled with query {query}")

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

        self.write_to_changelog(element=blockmodel_name, action='create', description=f"Schema written")

    def _delete_profile_report(self, blockmodel_name: str):
        """Delete the profile report from the OMF file when data has changed."""
        bm = self.get_element_by_name(blockmodel_name)

        if 'profile' in bm.metadata:
            del bm.metadata['profile']
