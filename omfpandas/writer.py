import json
import os
from pathlib import Path
from typing import Optional, Literal, Union

import omf
import pandas as pd
import ydata_profiling

from omfpandas import OMFPandasReader
from omfpandas.audit import ChangeMessage
from omfpandas.base import OMFPandas
from omfpandas.blockmodels.convert_blockmodel import df_to_blockmodel, blockmodel_to_df

from omfpandas.extras import _import_ydata_profiling, _import_pandera, _import_pandera_io
from omfpandas.utils.pandas_utils import parse_vars_from_expr
from omfpandas.utils.pandera_utils import DataFrameMetaProcessor, load_schema_from_yaml
from omfpandas.utils import log_timer

def get_username():
    try:
        return os.getlogin()
    except OSError:
        return os.environ.get('USER') or os.environ.get('USERNAME') or 'unknown_user'

PathLike = Union[str, Path, os.PathLike]

class OMFPandasWriter(OMFPandasReader):
    """A class to write pandas dataframes to an OMF file.

    Methods are named to align with CRUD
    create -> create a new element on the omf file
    read -> read a dataframe from an element (using OMFPandasReader inheritance)
    update -> update (part of) an existing element by providing a dataframe  # TODO
    delete -> delete an element from the omf file.  # TODO

    Attributes:
        filepath (Path): Path to the OMF file.
    """

    def __init__(self, filepath: PathLike):
        """Instantiate the OMFPandasWriter object.

        Args:
            filepath (Path): Path to the OMF file.
        """
        OMFPandas.__init__(self, filepath)
        self.user_id = get_username()

        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        if not filepath.exists():
            # log a message and create a new project
            project = omf.Project()
            project.name = filepath.stem
            project.description = f"OMF file created by OMFPandasWriter: {filepath.name}"
            self._logger.info(f"Creating new OMF file: {filepath}")
            self.project = project  # to enable the write_to_changelog method
            # create the audit record
            self.write_to_changelog(element='None', action='create', description=f"File created: {filepath}")
            # save the (now modified) project to the omf file
            self.persist_project()

        super().__init__(filepath)

    @log_timer()
    def create_blockmodel(self, blocks: pd.DataFrame, blockmodel_name: str,
                          pd_schema: Optional[Union[Path, dict]] = None,
                          allow_overwrite: bool = False):
        """Create an omf BlockModel from a dataframe.

        Only dataframes with centroid (x, y, z) and block dims (dx, dy, dz) indexes are supported.

        Args:
            blocks (pd.DataFrame): The dataframe to write to the BlockModel.
            blockmodel_name (str): The name of the BlockModel to write to. Use dot notation for composite (e.g., Composite.BlockModel).
            pd_schema (Optional[Union[Path, dict]]): The path to the Pandera schema file or a dict of the schema.
             Default is None.  If provided, the schema will be used to validate the dataframe before writing.
            allow_overwrite (bool): If True, overwrite the existing BlockModel. Default is False.

        Raises:
            ValueError: If the element retrieved is not a BlockModel.
        """
        composite_name = None
        full_blockmodel_name = blockmodel_name

        if '.' in blockmodel_name:
            composite_name, blockmodel_name = blockmodel_name.split('.', 1)

        calculation_map: dict = {}
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
                pd_schema = load_schema_from_yaml(pd_schema)
                # below line suffers from bug: https://github.com/unionai-oss/pandera/issues/1301
                # pd_schema = pa.DataFrameSchema.from_yaml(pd_schema)
                self._logger.info(f"Validating dataframe with schema: {pd_schema}")

            # validate the dataframe, which may modify it via coercion

            # add any calculated attributes in the schema
            dfmp: DataFrameMetaProcessor = DataFrameMetaProcessor(schema=pd_schema)
            calculation_map = dfmp.calculation_map
            blocks = dfmp.preprocess(blocks)
            blocks = dfmp.validate(blocks, return_calculated_columns=False)

            self._logger.info(f"Creating BlockModel from dataframe: {blockmodel_name}")
            bm = df_to_blockmodel(blocks, blockmodel_name)

            # persist the schema inside the omf file
            bm.description = pd_schema.description
            bm.metadata['pd_schema'] = pd_schema.to_json()
        else:
            self._logger.info(f"Creating BlockModel from dataframe: {blockmodel_name}")
            bm = df_to_blockmodel(blocks, blockmodel_name)

        if bm.name in [element.name for element in self.project.elements]:
            if not allow_overwrite:
                raise ValueError(f"BlockModel '{blockmodel_name}' already exists in the OMF file: {self.filepath}.  "
                                 f"If you want to overwrite, set allow_overwrite=True.")
            else:
                # remove the existing volume from the project
                volume_to_remove = [element for element in self.project.elements if element.name == bm.name][0]
                self.project.elements.remove(volume_to_remove)

        log_description: str = f"BlockModel written with {len(bm.attributes)} attributes"
        if composite_name:
            # get the composite if it exists or create it if it does not
            if composite_name not in [element.name for element in self.project.elements]:
                composite = omf.Composite(name=composite_name)
                self.project.elements.append(composite)
            else:
                composite = self.get_element_by_name(composite_name)
            composite.elements.append(bm)
            log_description += f" in composite {composite_name}"
        else:
            self.project.elements.append(bm)

        # create the audit record
        self.write_to_changelog(element=bm.name, action='create', description=log_description)
        # write the omf project to file
        self.persist_project()

        # write the calculated variables to the omf block model metadata
        if pd_schema is not None:
            self.create_calculated_blockmodel_attributes(full_blockmodel_name, calc_definitions=calculation_map)

    def create_calculated_blockmodel_attributes(self, blockmodel_name: str, calc_definitions: dict[str, str]):
        """Create a calculated attribute for a BlockModel.

        Calculated attributes reduce storage space by storing the calculation expression instead of the data.
        When the attribute is accessed, the expression is evaluated and the result returned.
        The calculation expression must be a valid pandas expression, and is stored in the metadata of the
        blockmodel object.  Only OMF2 supports this feature.

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
                col_schema = pa.io.from_json(bm.metadata['pd_schema']).columns.get(attr_name)
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
                                    description=f"Calculated attribute [{attr_name}] added with expression {expr}")

        self.persist_project()

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

    def persist_project(self):
        """Persist the omf project to file and reload the project property"""
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
        from omfpandas.blockmodels.attributes import series_to_attribute

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

    def blockmodel_to_parquet(self, blockmodel_name: str, out_path: Optional[Path] = None,
                              variables: Optional[list[str]] = None,
                              allow_overwrite: bool = False):
        """Convert blockmodel to a Parquet file.

        Args:
            blockmodel_name (str): The BlockModel element to convert.
            out_path (Optional[Path]): The path to the Parquet file to write. If None, a file with the blockmodel name is
            created.
            variables (Optional[list[str]]): The variables to include in the DataFrame. If None, all variables are included.
            allow_overwrite (bool): If True, overwrite the existing Parquet file. Default is False.

        Raises:
            FileExistsError: If the file already exists and allow_overwrite is False.
        """
        bm = self.get_element_by_name(blockmodel_name)
        if out_path is None:
            out_path = Path(f"{blockmodel_name}.parquet")
        if out_path.exists() and not allow_overwrite:
            raise FileExistsError(
                f"File already exists: {out_path}. If you want to overwrite, set allow_overwrite=True.")
        df: pd.DataFrame = blockmodel_to_df(blockmodel=bm, variables=variables)
        df.to_parquet(out_path)

    def blockmodel_to_orc(self, blockmodel_name: str, out_path: Optional[Path] = None,
                          variables: Optional[list[str]] = None,
                          allow_overwrite: bool = False):
        """Convert blockmodel to an ORC file.

        Args:
            blockmodel_name (str): The BlockModel element to convert.
            out_path (Optional[Path]): The path to the ORC file to write. If None, a file with the blockmodel name is
            created.
            variables (Optional[list[str]]): The variables to include in the DataFrame. If None, all variables are included.
            allow_overwrite (bool): If True, overwrite the existing ORC file. Default is False.

        Raises:
            FileExistsError: If the file already exists and allow_overwrite is False.
        """
        bm = self.get_element_by_name(blockmodel_name)

        if out_path is None:
            out_path = Path(f"{blockmodel_name}.orc")
        if out_path.exists() and not allow_overwrite:
            raise FileExistsError(
                f"File already exists: {out_path}. If you want to overwrite, set allow_overwrite=True.")
        is_tensor: bool = True if bm.__class__.__name__ == 'TensorGridBlockModel' else False
        df: pd.DataFrame = blockmodel_to_df(blockmodel=bm, variables=variables)
        df.to_orc(out_path)
