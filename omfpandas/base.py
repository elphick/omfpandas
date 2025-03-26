import json
import json
import logging
import os
import tempfile
import webbrowser
from abc import ABC
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union, Any

import omf
import pandas as pd

if TYPE_CHECKING:
    from omf import Project
    from omfpandas.blockmodels.geometry import RegularGeometry, TensorGeometry

SUPPORTED_BM_TYPES = ['RegularBlockModel', 'TensorGridBlockModel']

PathLike = Union[str, Path, os.PathLike]


class OMFPandas(ABC):

    def __init__(self, filepath: PathLike):
        """Instantiate the OMFPandas object.

        Args:
            filepath (Path): Path to the OMF file.

        Raises:
            FileNotFoundError: If the OMF file does not exist.
            ValueError: If the file is not an OMF file.
        """
        self._logger = logging.getLogger(__class__.__name__)

        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        if not filepath.suffix == '.omf':
            raise ValueError(f'File is not an OMF file: {filepath}')
        self.filepath: Path = filepath
        self.project: Optional[Project] = None
        if filepath.exists():
            self.project = omf.load(str(filepath))

    def __repr__(self):
        res: str = f"OMF file({self.filepath})"
        res += f"\nElements: {self.element_types}"
        return res

    def __str__(self):
        res: str = f"OMF file({self.filepath})"
        res += f"\nElements: {self.element_types}"
        return res

    @property
    def element_types(self) -> Optional[dict[str, Any]]:
        """Dictionary of elements keyed by name

        In the special case of a composite element, the key will be the composite name and
        the value will be a dictionary of child elements keyed by name.

        """
        _elements = self.project.elements if self.project else []
        if _elements:
            element_dict = {}
            for e in _elements:
                if hasattr(e, 'elements') and e.elements:
                    element_dict[e.name] = {child.name: child.__class__.__name__ for child in e.elements}
                else:
                    element_dict[e.name] = e.__class__.__name__
            return element_dict
        else:
            return {}

    @property
    def blockmodel_attributes(self) -> Optional[dict[str, Any]]:
        """Attributes for blockmodel elements, keyed by element name, including composites."""
        elements = {}
        for el in self.project.elements:
            if el.__class__.__name__ in ['TensorGridBlockModel', 'RegularBlockModel']:
                elements[el.name] = [a.name for a in el.attributes]
            elif hasattr(el, 'elements') and el.elements:
                composite_elements = {}
                for child in el.elements:
                    if child.__class__.__name__ in ['TensorGridBlockModel', 'RegularBlockModel']:
                        composite_elements[child.name] = [a.name for a in child.attributes]
                if composite_elements:
                    elements[el.name] = composite_elements

        return elements if elements else {}

    @property
    def changelog(self) -> Optional[pd.DataFrame]:
        """Return the change log as a DataFrame."""
        if 'changelog' not in self.project.metadata:
            return None
        return pd.DataFrame([json.loads(msg) for msg in self.project.metadata['changelog']])

    def get_element_by_name(self, element_name: str):
        """Get an element by its name.

        :param element_name: The name of the element to retrieve. Use dot notation for composite (e.g., Composite.BlockModel).
        :return:
        """
        composite_name = None
        if '.' in element_name:
            composite_name, element_name = element_name.split('.', 1)

        if composite_name:
            composite = [e for e in self.project.elements if e.name == composite_name]
            if not composite:
                raise ValueError(f"Composite '{composite_name}' not found in the OMF file: {self.filepath.name}.")
            composite = composite[0]
            element = [e for e in composite.elements if e.name == element_name]
            if not element:
                raise ValueError(f"Element '{element_name}' not found in composite '{composite_name}'.")
        else:
            element = [e for e in self.project.elements if e.name == element_name]
            if not element:
                raise ValueError(f"Element '{element_name}' not found in the OMF file: {self.filepath.name}.")
            elif len(element) > 1:
                raise ValueError(f"Multiple elements with the name '{element_name}' found in the OMF file: "
                                 f"{self.filepath.name}")

        return element[0]

    def get_element_attribute_names(self, element_name: str) -> list[str]:
        """Get the attribute names of an element.

        :param element_name: The name of the element to retrieve.  Use dot notation for elements in a composite.
        :return:
        """
        element = self.get_element_by_name(element_name)
        return [attr.name for attr in element.attributes]

    def get_bm_geometry(self, blockmodel_name: str) -> Union['RegularGeometry', 'TensorGeometry']:
        """Get the geometry of a BlockModel.

        Args:
            blockmodel_name (str): The name of the BlockModel to retrieve.

        Returns:
            TensorGeometry: The geometry of the BlockModel.
        """
        bm = self.get_element_by_name(blockmodel_name)
        if bm.__class__.__name__ == 'TensorGridBlockModel':
            from omfpandas.blockmodels.geometry import TensorGeometry
            return TensorGeometry.from_element(bm)
        elif bm.__class__.__name__ == 'RegularBlockModel':
            from omfpandas.blockmodels.geometry import RegularGeometry
            return RegularGeometry.from_element(bm)
        else:
            raise ValueError(
                f"Element '{blockmodel_name}' is not a supported BlockModel in the OMF file: {self.filepath}")

    def view_block_model_profile(self, blockmodel_name: str, query: Optional[str] = None):
        """View the profile of a BlockModel in the default web browser.

        Args:
            blockmodel_name (str): The name of the BlockModel to profile.
            query (str): A query defining the subset of the BlockModel.
        """

        el = self.get_element_by_name(blockmodel_name)
        filter_key: str = query if query else 'no_filter'

        if el.metadata.get('profile') is None:
            raise ValueError(f"BlockModel '{blockmodel_name}' has not been profiled.  "
                             f"Please run 'profile_blockmodel' first.")

        # JSON string containing the profile report
        profile_html: str = el.metadata['profile'][filter_key]

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
            temp_file.write(profile_html.encode('utf-8'))
            temp_file_path = temp_file.name

        # Open the temporary file in the default web browser
        webbrowser.open(f"file://{temp_file_path}")
